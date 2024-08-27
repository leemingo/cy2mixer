import torch
import torch.nn as nn
import torch.nn.functional as F

class poly_INR(nn.Module):
    def __init__(self,
                 input_dim : int,
                 num_class : int, 
                 hidden_dim : int, 
                 num_layers : int,
                 num_embedding_layers : int,
                 activation_fn : nn.Module = nn.LeakyReLU(negative_slope=0.2)
                 ):
        super(poly_INR, self).__init__()
        self.num_layers = num_layers
        self.class_embedding = nn.Sequential()
        self.class_embedding.append(nn.Linear(num_class,hidden_dim))
        for _ in range(num_embedding_layers - 1):
            self.class_embedding.append(nn.Linear(hidden_dim, hidden_dim))
        self.latent = nn.parameter.Parameter(torch.ones([hidden_dim]),requires_grad=True) 
        #self.projectors = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]) 
        self.A_linears = nn.ModuleList([nn.Linear(1,input_dim) for _ in range(num_layers)]) 
        self.activation_fn = activation_fn

        for i in range(self.num_layers):
            setattr(self, f'FC_{i:d}', FCLayer_aff(hidden_dim, hidden_dim, nn.ReLU(inplace=True)))

    def reset_parameters(self):
        for lin in self.projectors:
            lin.reset_parameters()
        for lin in self.A_linears:
            lin.reset_parameters()

    def forward(self, 
                x : torch.tensor,
                c : torch.tensor,
                coord : torch.tensor
                ):
        x0 = x 

        coords_CAM = coord                       # [N, 2] 
        coords_CAM = coords_CAM.view(1,-1, 1, 2) # CAM mlp에 쓸수있게 변경
        
        W = self.class_embedding(c) + self.latent
        W = W.reshape(-1,1) #[hidden,1]
        A = self.A_linears[0](W) 
        
        poly = torch.matmul(x0,A.T) 
        x = poly 
        for i in range(self.num_layers-1):
            #x = self.activation_fn(self.projectors[i](x))
            CAM = getattr(self, f'FC_{i:d}')
            x = self.activation_fn(CAM(x, coords_CAM))
            A = self.A_linears[i+1](W) 
            poly = torch.matmul(x0,A.T) 
            x = x*poly 
        CAM = getattr(self, f'FC_{self.num_layers-1:d}')
        x = self.activation_fn(CAM(x, coords_CAM))
        #x = self.activation_fn(self.projectors[-1](x)) 
        return x
    
# cam을 위한 layer
class FCLayer_aff(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.w = 32
        self.h = 32
        self.linear = nn.Linear(in_features, out_features)
        self.affine = nn.Parameter(torch.cat([torch.ones(1,1,self.w, self.h),torch.zeros(1,1,self.w, self.h)], dim=1))
        self.act = act

    def forward(self, input, coord):

        #N, _ = input.shape
        # print('shape of input', input.shape)
        # print('shape of coord', coord.shape) #[1, N, 1, 3]
        output = self.linear(input)
        output = F.instance_norm(output.unsqueeze(0))
        output = output.squeeze(0)
        #print('shape of output', output.shape) # [N, hidden]

        affine = nn.functional.grid_sample(self.affine, coord, padding_mode='border', align_corners=True).view(2,-1,1)

        #print('shape of affine', affine.shape) # [2, N, 1]

        output = output*affine[0]+affine[1]  # [N, hidden]
        output = self.act(output)
        return output