import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLinear(nn.Module):
    def __init__(self,feat_dim,out_dim) -> None:
        super().__init__()
        self.weight =nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.sigma = 1.0   
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
     
    
    
    def forward(self,x):
        x = F.normalize(x,dim = -1)
        norm_weight = F.normalize(self.weight, dim = -1)
        out  = F.linear(x,norm_weight)
        return self.sigma * out

class SingleLinear(nn.Module):
    def __init__(self,feat_dim,out_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim,feat_dim)
        self.weight =nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.sigma = 1.0   
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
     
    
    
    def forward(self,x):
        x = self.proj(x)
        x = F.relu(x)
        x = F.normalize(x,dim = -1)
        norm_weight = F.normalize(self.weight, dim = -1)
        out  = F.linear(x,norm_weight)
        return self.sigma * out