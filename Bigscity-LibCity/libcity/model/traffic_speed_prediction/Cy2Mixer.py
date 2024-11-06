from logging import getLogger

import torch.nn as nn
from torch.nn import functional as F
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_self_loops
from torch_geometric.nn.conv import GCNConv

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, num_node, kernel_size, conv_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.conv_type = conv_type
        
        if conv_type =='conv':
            if kernel_size == 1:
                self.spatial_proj = nn.Conv2d(num_node, num_node, kernel_size=kernel_size)
            else:
                self.spatial_proj = nn.Conv2d(
                    num_node, num_node, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
                )    
        elif conv_type == 'mpnn':
            self.spatial_proj = GCNConv(d_ffn, d_ffn)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        
    def forward(self, x, residual=None, cirmat=None):
        # x= [B,T,N,D]
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        if self.conv_type == 'conv':
            x = self.spatial_proj(v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        elif self.conv_type == 'mpnn':
            edge_index = dense_to_sparse(torch.Tensor(cirmat))[0] 
            x = self.spatial_proj(v, edge_index)
                
        if residual is not None:
            out = (x + residual) * u
        else:
            out = u * x
        return out

class Cy2MixerBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_node, kernel_size, conv_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        # setattr(self, f'channel_proj1', FCLayer_aff(d_model, d_ffn * 2, nn.ReLU(inplace=True)))
        # setattr(self, f'channel_proj2', FCLayer_aff(d_ffn, d_model, nn.ReLU(inplace=True)))
        self.latent = nn.Linear(d_model, 2)
        self.sgu = SpatialGatingUnit(d_ffn, num_node, kernel_size, conv_type)
        self.channel_proj_tinyatt = nn.Conv2d(d_ffn, 3 * d_ffn, kernel_size=1)
        self.out_proj_tinyatt = nn.Conv2d(d_ffn, d_ffn, kernel_size=1)

    def forward(self, x, residual=None, cirmat=None, use_tinyatt=False):
        residual = x
        x = self.norm(x)
        if use_tinyatt:
            qkv = self.channel_proj_tinyatt(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # [B,  N, T, 3*D]
            q, k, v = qkv.chunk(3, dim= -1) #[B, N, T, D] * 3
            w = q @ k.transpose(-1, -2)
            w = w.softmax(dim=-1)
            gate_res = w @ v
            gate_res = self.out_proj_tinyatt(gate_res.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        coords = self.latent(x)
        # x = self.channel_proj1(x, coords)
        x = F.gelu(self.channel_proj1(x))
        if use_tinyatt:
            x = self.sgu(x, residual=gate_res, cirmat=cirmat)
        else:
            x = self.sgu(x, residual=None, cirmat=cirmat)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class Cy2Mixer_layer(nn.Module):
    def __init__(
        self, d_model, d_ffn, seq_len, num_node, output_dim=1, bias=False, use_tinyatt = False,
        dropout = 0.1, tgu_kernel_size = 3
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.seq_len = seq_len
        self.num_node = num_node
        self.output_dim = output_dim
        self.use_tinyatt = use_tinyatt
        self.tgu_kernel_size = tgu_kernel_size
        
        
        self.tgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.seq_len, kernel_size= self.tgu_kernel_size, conv_type = 'conv')
        self.sgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.num_node, kernel_size=1, conv_type = 'mpnn')
        self.cgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.num_node, kernel_size=1, conv_type = 'mpnn')
        self.channel_proj_out = nn.Conv2d(3 * self.d_model, self.d_model, kernel_size=1)
        # self.channel_proj_out = nn.Conv2d(2 * self.d_model, self.d_model, kernel_size=1)
            
        
    def forward(
        self,
        x,
        cirmat_A=None,
        cirmat_C=None,
    ):
        # x=[B,T,N,D]
        residual = x
        x_t = self.tgu(x.permute(0, 2, 1, 3), use_tinyatt=self.use_tinyatt).permute(0, 2, 1, 3)
        x_t = self.norm1(x_t)
        x_t = self.dropout1(x_t)
        
        x_s = self.sgu(x, residual=residual, cirmat=cirmat_A, use_tinyatt=self.use_tinyatt)
        x_s = self.norm2(x_s)
        x_s = self.dropout2(x_s)
        
        x_c = self.cgu(x, residual=residual, cirmat=cirmat_C, use_tinyatt=self.use_tinyatt)
        x_c = self.norm3(x_c)
        x_c = self.dropout3(x_c)
          
        # x=[B,T,N,3*D]
        x = torch.cat([x_t, x_s, x_c], dim=-1)
        # x = torch.cat([x_t, x_s], dim=-1)
        x = self.channel_proj_out(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) # x=[B,T,N,D]
        # x = self.norm(x)
        out = x + residual
        return out



class Cy2Mixer(AbstractTrafficStateModel):
    def __init__(self,config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        self.num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')
        self.A_mx = self.data_feature.get('adj_mx')
        self.A_mx[self.A_mx != 0] = 1
        self.A_mx = torch.Tensor(self.A_mx).long().to(self.device)
        self.A_mx.fill_diagonal_(0)
        self.C_mx = torch.Tensor(self.data_feature.get('cycle_mx')).long()
        # self.C_mx, _ = add_self_loops(self.C_mx)
        self.C_mx = (
            to_dense_adj(
                self.C_mx, max_num_nodes= self.num_nodes
            )
            .squeeze(0)
        ).to(self.device)
        self.edge_index = dense_to_sparse(self.A_mx)[0]
        self.cycle_index = dense_to_sparse(self.C_mx)[0]
        

        self.in_steps = config.get('input_window', 12)
        self.out_steps = config.get('output_window', 12)
        self.steps_per_day = config.get('steps_per_day', 288)
        self.input_dim = config.get('input_dim', 3)
        self.output_dim = config.get('output_dim', 1)
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 0)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)
        self.tgu_kernel_size = config.get('tgu_kernel_size', 3)
        self.use_tinyatt = config.get("use_tinyatt", True)

        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )

        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.use_mixed_proj = config.get('use_mixed_proj', True)
        # GPU_ID = gpu_num
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
        # self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
            
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.encoder_blocks = nn.ModuleList(
                [
                    Cy2Mixer_layer(
                        self.model_dim,
                        self.model_dim,
                        self.in_steps,
                        self.num_nodes,
                        self.output_dim,
                        use_tinyatt = self.use_tinyatt,
                        dropout = self.dropout,
                        tgu_kernel_size = self.tgu_kernel_size
                    )
                    for _ in range(self.num_layers)
                ]
            )
    
        
    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
            
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(
                x,
                cirmat_A = self.A_mx,
                cirmat_C = self.C_mx,
            )

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
    
    def predict(self, batch):
        return self.forward(batch['X'])
    
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)