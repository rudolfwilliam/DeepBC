"""Generic conditional flow class without specified archtitecture: to be implemented by subclasses."""

from torch.optim import Adam
import pytorch_lightning as pl
from .structural_equation import StructuralEquation


class GCondFlow(pl.LightningModule, StructuralEquation):
    def __init__(self, name, lr=1e-6, verbose=False):
        super().__init__()
        self.name = name
        self.lr = lr
        self.verbose = verbose

    def forward(self, x, x_pa):
        return self.flow(x, x_pa)
    
    def encode(self, x, x_pa):
        return self.flow.inverse(x, x_pa)
    
    def decode(self, u, x_pa):
        return self.flow(u, x_pa)
    
    def training_step(self, train_batch, batch_idx):
        x, x_pa = train_batch
        loss = self.flow.forward_kld(x, x_pa)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, x_pa = val_batch 
        loss = self.flow.forward_kld(x, x_pa)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def inverse_and_log_det(self, x, x_pa):
        return self.flow.inverse_and_log_det(x, x_pa)
    
    def on_train_epoch_end(self):
        if self.verbose:
            print(f"train_loss = {self.trainer.callback_metrics['train_loss']}, val_loss = {self.trainer.callback_metrics['val_loss']}")
            