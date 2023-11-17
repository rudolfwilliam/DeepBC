from celeba.data.meta_data import attrs
import torch

def identity(x, x_ast):
    return torch.mean((torch.cat([x[attr] for attr in attrs], dim=1) - 
                      torch.cat([x_ast[attr] for attr in attrs], dim=1))**2)

def causal(x_ast, scm):
    return torch.mean(torch.cat([scm.encode(**x_ast)[attr] for attr in attrs], dim=1)**2)
