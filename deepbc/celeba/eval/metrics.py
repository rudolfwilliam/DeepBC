from celeba.data.meta_data import attrs
import torch

def obs(x, x_ast):
    return torch.mean((torch.cat([x[attr] for attr in attrs], dim=1) - 
                      torch.cat([x_ast[attr] for attr in attrs], dim=1))**2)

def plausible(x_ast, scm):
    return torch.mean(torch.cat([scm.encode(**x_ast)[attr] for attr in attrs], dim=1)**2)

def causal(x, x_ast, scm):
    return torch.mean((torch.cat([scm.encode(**x)[attr] for attr in attrs for attr in attrs], dim=1) - 
                       torch.cat([scm.encode(**x_ast)[attr] for attr in attrs for attr in attrs], dim=1))**2)
