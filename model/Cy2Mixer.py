import numpy as np
import os 
import pickle
import torch.nn as nn
import torch
from torchinfo import summary
from layers import Cy2Mixer_layer
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
from torch_geometric.typing import SparseTensor


class Cy2Mixer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        gpu_num = 0,
        dataset = 'pems04',
        use_tinyatt = True,
        tgu_kernel_size = 3
            ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        GPU_ID = gpu_num
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.use_tinyatt = use_tinyatt
        with open(f'../data/{self.dataset.upper()}/{self.dataset}_A_mx.pkl', 'rb') as f:
            self.A_mx = pickle.load(f)
        self.A_mx = torch.Tensor(self.A_mx).to(self.device)
        self.C_mx = torch.Tensor(np.load(f'../data/{self.dataset.upper()}/{self.dataset}_cycle.npy')).long()
        self.C_mx = (
            to_dense_adj(
                self.C_mx, max_num_nodes= num_nodes
            )
            .squeeze(0)
        ).to(self.device)
        self.edge_index = dense_to_sparse(self.A_mx)[0]
        self.cycle_index = dense_to_sparse(self.C_mx)[0]
        
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
            
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        #MH
        self.encoder_blocks = nn.ModuleList(
                [
                    Cy2Mixer_layer(
                        self.model_dim,
                        self.model_dim,
                        self.in_steps,
                        self.num_nodes,
                        self.output_dim,
                        use_tinyatt = self.use_tinyatt,
                        dropout = dropout,
                        tgu_kernel_size = tgu_kernel_size
                    )
                    for _ in range(num_layers)
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

if __name__ == "__main__":
    model = Cy2Mixer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
