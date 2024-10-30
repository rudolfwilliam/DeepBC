import torch
from pytorch_lightning import Trainer
from ...utils import generate_checkpoint_callback, generate_early_stopping_callback

def train_vae(vae_img, train_set, val_set, default_root_dir, config):
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", gradient_clip_val=config["gradient_clip_val"], callbacks=[generate_checkpoint_callback(config["name"], config["ckpt_path"]), 
                      generate_early_stopping_callback(patience=config["patience"])], default_root_dir=default_root_dir, max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(vae_img, train_data_loader, val_data_loader)
