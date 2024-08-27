import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import GCNConv,  GATConv, GATv2Conv
from torch_geometric.utils import dense_to_sparse

from poly_inr import FCLayer_aff

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
        # self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        # self.channel_proj2 = nn.Linear(d_ffn, d_model)
        setattr(self, f'channel_proj1', FCLayer_aff(d_model, d_ffn * 2, nn.ReLU(inplace=True)))
        setattr(self, f'channel_proj2', FCLayer_aff(d_ffn, d_model, nn.ReLU(inplace=True)))
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
        x = self.channel_proj1(x, coords)
        # x = F.gelu(self.channel_proj1(x))
        if use_tinyatt:
            x = self.sgu(x, residual=gate_res, cirmat=cirmat)
        else:
            x = self.sgu(x, residual=None, cirmat=cirmat)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class Cy2MixerBlock_aff(nn.Module):
    def __init__(self, d_model, d_ffn, num_node, kernel_size, conv_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
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
        # self.cgu = Cy2MixerBlock(self.d_model, self.d_ffn, self.num_node, kernel_size=1, conv_type = 'mpnn')
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

