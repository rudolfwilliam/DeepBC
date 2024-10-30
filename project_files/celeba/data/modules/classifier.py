"""Obtain continuous labels for attributes via classifier."""


from torch.optim import Adam
import torch.nn as nn
import pytorch_lightning as pl
import torch

class Classifier(pl.LightningModule):
    def __init__(self, attr, n_chan=[3, 8, 16, 32, 32, 64, 1]):
        super().__init__()
        self.attr = attr
        self.conv = nn.Sequential(            
            nn.Conv2d(in_channels=n_chan[0], out_channels=n_chan[1], kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=n_chan[1], out_channels=n_chan[2], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=n_chan[2], out_channels=n_chan[3], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_chan[3], out_channels=n_chan[4], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ln = nn.Sequential(
            nn.Linear(in_features=n_chan[4], out_features=n_chan[5]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=n_chan[5], out_features=n_chan[6]),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32)
        x = self.ln(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, attrs = batch
        y = attrs[:, self.attr]
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, attrs = batch
        y = attrs[:, self.attr]
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1))
        self.log("val_loss", loss) 
        print("validation loss: ", loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer