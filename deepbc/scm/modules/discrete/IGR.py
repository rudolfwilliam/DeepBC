"""Invertible Gaussian Reparameterization (IGR) module."""

from scm.modules import StructuralEquation
from torch.nn.functional import one_hot
import torch.distributions as dist
import torch
import pytorch_lightning as pl
import torch.nn as nn

class IGR(pl.LightningModule, StructuralEquation):
    def __init__(self, name, parents, num_classes=2, tau=1e-1, delta=1, lr=1e-2):
        super(IGR, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.lr = lr
        # learnable mean and log-variance
        self.table = ParamTable(len(parents), num_classes)
        self.tau = tau
        self.delta = torch.tensor(delta)
    
    def encode_relaxed(self, x, x_pa):
        mu, logvar = self.table(x, x_pa)
        u = torch.zeros(x.shape[0], self.num_classes-1)
        last = x[:, self.num_classes-1]
        Z_nolast = self.delta * (1 - last)/last
        for i in range(self.num_classes - 1):
            u[:, i] = (self.tau*(torch.log(x[:, i]) + torch.log(self.delta + Z_nolast)) - mu[i])/torch.exp(0.5 * logvar[i])
        return u
    
    def encode(self, x, x_pa):
        # one hot encode and smoothen x
        idx = x.squeeze()
        eps = torch.tensor(1e-1, dtype=torch.float32)
        x = torch.tensor(one_hot(x, num_classes=self.num_classes), dtype=torch.float32).squeeze()
        x[range(len(idx)), idx] = x[range(len(idx)), idx] - eps
        # Generate all indices along the specified dimension
        all_idx = torch.arange(x.size(1))
        # Use boolean indexing to exclude the specified indices
        idx_to_select = all_idx[~torch.tensor(idx)]
        # Use advanced indexing to select elements
        x[range(len(idx_to_select)), idx_to_select] = x[range(len(idx_to_select)), idx_to_select] + eps/(self.num_classes - 1)
        # compute u
        u = self.encode_relaxed(x, x_pa)
        return u, x
    
    def decode(self, u, x_pa):
        mu, logvar = self.table(u, x_pa)
        Y = mu + torch.exp(0.5 * logvar) * u
        Y = torch.cat([Y, torch.log(self.delta.repeat(Y.shape[0], 1))*self.tau], dim=1)
        states = torch.stack([torch.exp(Y[:, i]/self.tau) for i in range(self.num_classes)], dim=1)
        Z = torch.sum(states, dim=1)
        return states/Z.unsqueeze(1)
    
    def forward(self, u, x_pa):
        return self.decode(u, x_pa)
    
    def step(self, batch, batch_idx, mode='train'):
        # maximum likelihood estimation
        x, x_pa = batch
        # compute loss via change of variables formula
        def decode_no_pa(u):
            return self.decode(u, x_pa)[:, :-1]
        _, x_hat = self.encode(x, x_pa)
        x_hat = x_hat.detach().requires_grad_()
        u = self.encode_relaxed(x_hat, x_pa)
        loss = -dist.Normal(0, 1).log_prob(u) + torch.abs(torch.det(torch.diagonal(torch.autograd.functional.jacobian(decode_no_pa, u), dim1=0, dim2=2).permute(2, 0, 1))).unsqueeze(-1)
        loss = loss.mean()
        self.log(mode + "_loss", loss, on_step=False, on_epoch=True)
        print("params: ", self.table.param_dict['[]'])
        print(round(loss.item(), 3))
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='train')    
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='val')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class ParamTable(nn.Module):
    def __init__(self, num_input_states, num_classes=2):
        super(ParamTable, self).__init__()
        # Create a dict to store learnable parameters
        self.param_dict = nn.ParameterDict()
        self.num_classes = num_classes
        # Create parameters for each combination of input states
        if num_input_states == 0:
            self.param_dict['[]'] = nn.Parameter(torch.zeros(2*(num_classes - 1)), requires_grad=True)
        else:
            for i in range(num_classes ** num_input_states):
                state_key = tuple(int(bit) for bit in format(i, f'0{num_input_states}b'))
                param_name = f'param_{state_key}'
                self.param_dict[param_name] = nn.Parameter(torch.zeros(2*(num_classes - 1)), requires_grad=True) 

    def forward(self, x, x_pa):
        return (torch.stack([self.param_dict[str(x_pa[i].tolist())][:(self.num_classes - 1)] for i in range(x.shape[0])], dim=0), 
                torch.stack([self.param_dict[str(x_pa[i].tolist())][(self.num_classes - 1):] for i in range(x.shape[0])], dim=0))
    