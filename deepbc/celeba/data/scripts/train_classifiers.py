from pytorch_lightning import Trainer
from utils import generate_checkpoint_callback, generate_early_stopping_callback
from celeba.data.modules import Classifier
from celeba.data.datasets import load_data
from celeba.data.meta_data import attrs, attr2int
from json import load
import torch

def train_classifier(classifier, attr, train_set, val_set, config, default_root_dir):
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", 
                      callbacks=[generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"]), 
                                 generate_early_stopping_callback(patience=config["patience"])], 
                      default_root_dir=default_root_dir, max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(classifier, train_data_loader, val_data_loader)

def main(ckpt_path="./celeba/scm/trained_models/config/", default_root_dir="./celeba/scm"):
    torch.manual_seed(42)
    # Load the data
    data = load_data()
    config = load(open(ckpt_path, "r"))
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1-config["train_val_split"]])
    # initialize models
    for attr in attrs:
        clsfier = Classifier(attr=attr2int[attr])
        train_classifier(clsfier, attr, train_set, val_set, config, default_root_dir)
    print("done.")

if __name__ == "__main__":
    main()
