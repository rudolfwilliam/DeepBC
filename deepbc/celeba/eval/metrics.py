from deepbc.celeba.data.meta_data import attrs
import torch

def obs(x, x_ast, mode="l2"):
    if mode == "l2":
        return torch.mean((torch.cat([x[attr] for attr in attrs], dim=1) - 
                           torch.cat([x_ast[attr] for attr in attrs], dim=1))**2)
    if mode == "l1":
        return torch.mean(torch.abs(torch.cat([x[attr] for attr in attrs], dim=1) - 
                                     torch.cat([x_ast[attr] for attr in attrs], dim=1)))

def plausible(x_ast, scm):
    log_dets = []
    for attr in attrs:
        if scm.graph_structure[attr] == []:
            log_dets.append(scm.models[attr].inverse_and_log_det(x_ast[attr], torch.tensor([]))[1])
        else:
            log_dets.append(scm.models[attr].inverse_and_log_det(x_ast[attr], torch.concat([x_ast[attr_pa] for attr_pa in scm.graph_structure[attr]], dim=1))[1])
    # simple change of variables
    return torch.mean(torch.cat([scm.encode(**x_ast)[attr] for attr in attrs], dim=1)**2)/2 - \
           torch.mean(torch.tensor(log_dets))

def causal(x, x_ast, scm, mode="l2"):
    if mode == "l2":
        return torch.mean((torch.cat([scm.encode(**x)[attr] for attr in attrs], dim=1) - 
                           torch.cat([scm.encode(**x_ast)[attr] for attr in attrs], dim=1))**2)
    if mode == "l1":
        return torch.mean(torch.abs(torch.cat([scm.encode(**x)[attr] for attr in attrs], dim=1) - 
                                     torch.cat([scm.encode(**x_ast)[attr] for attr in attrs], dim=1)))
