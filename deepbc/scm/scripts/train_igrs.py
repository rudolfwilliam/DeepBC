from utils import generate_checkpoint_callback, generate_early_stopping_callback
from pytorch_lightning import Trainer
from data.datasets import SelectAttributesTransform
import torch

def train_igr(igr, config, data_class, graph_structure, attrs, default_root_dir, **kwargs):
    transform = SelectAttributesTransform(attrs.index(igr.name), [attrs.index(attr_pa) for attr_pa in graph_structure[igr.name]])
    # load the data (with discrete labels)
    data = data_class(transform=transform, **kwargs)
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", callbacks=[generate_checkpoint_callback(igr.name + "_igr", config["ckpt_path"]), 
                                                                                      generate_early_stopping_callback(patience=config["patience"])], 
                                                                                                                       default_root_dir=default_root_dir, 
                                                                                                                       max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(igr, train_data_loader, val_data_loader)
    