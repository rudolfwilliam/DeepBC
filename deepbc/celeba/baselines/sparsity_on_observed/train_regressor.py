"""Define sparsity in terms of observed variables rather than latents as done in backtracking."""

import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
from data.datasets import SelectAttributesTransform
from celeba.data.meta_data import attrs, graph_structure
from celeba.data.datasets import CelebaContinuous
from utils import generate_checkpoint_callback, generate_early_stopping_callback

class Regressor(pl.LightningModule):
    """Simple model that regresses one attribute from all others."""
    def __init__(self, ckpt_path, name="beard"):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        # simple architecture
        self.ln1 = nn.Linear(len(attrs) - 1, 10)
        self.relu1 = nn.ReLU()
        self.ln2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.ln3 = nn.Linear(10, 1)
    
    def load_parameters(self):
        # load pre-trained model for first file name starting with model name
        file_name = next((file for file in os.listdir() if file.startswith(self.name)), None)
        self.model.load_state_dict(torch.load(self.ckpt_path + file_name, map_location=torch.device('cpu'))["state_dict"])
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.relu1(self.ln1(x))
        x = self.relu2(self.ln2(x))
        x = self.ln3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
def main(attr="beard", patience=2, max_epochs=100, train_val_split=0.8, batch_size_train=128, ckpt_path="./celeba/baselines/sparsity_on_x/trained_models/checkpoints/"):
    """Train classifier on observed variables."""
    # initialize model
    regressor = Regressor(ckpt_path, name=attr)
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto", max_epochs=max_epochs, callbacks=[generate_checkpoint_callback("beard", ckpt_path), 
                                                                                                                generate_early_stopping_callback(patience=patience)], 
                                                                                                                default_root_dir="./celeba/baselines/sparsity_on_x/")
    # load the data (with continuous labels)
    data = CelebaContinuous(cont_attr_path="./celeba/data/predictions/preds.pt", transform=SelectAttributesTransform(attrs.index(attr), [attrs.index(attr_) for attr_ in attrs if attr_ != attr]))
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [train_val_split, 1 - train_val_split])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    # train model
    trainer.fit(regressor, train_data_loader, val_data_loader)
    print("done.")

if __name__ == "__main__":
    main()
    