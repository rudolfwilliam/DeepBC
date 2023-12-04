"""Invertible Gaussian Reparameterization (IGR) module."""

from scm.modules import StructuralEquation
import torch
import pytorch_lightning as pl

class IGR(pl.LightningModule, StructuralEquation):
    def __init__(self, name, num_classes, tau=1e-1, delta=1, lr=1e-6):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.lr = lr
        # learnable mean and log-variance
        self.mu = pl.Parameter(torch.zeros(self.num_classes - 1))
        self.logvar = pl.Parameter(torch.zeros(self.num_classes - 1))
        self.tau = tau
        self.delta = delta
    
    def encode(self, x, x_pa):
        u = torch.zeros(x.shape[0], self.num_classes)
        last = x[:, self.num_classes]
        Z = self.delta * (1 - last)/last
        # TODO: finish this
    
    def decode(self, u, x_pa):
        Y = self.mu + torch.exp(0.5 * self.logvar) * u
        Z = sum([torch.exp(-self.tau * Y[:, i]) for i in range(self.num_classes - 1)])
        return torch.cat([Y[:, i] / Z for i in range(self.num_classes - 1)], dim=1)
