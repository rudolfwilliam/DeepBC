from pytorch_lightning import Trainer
from utils import generate_checkpoint_callback, generate_early_stopping_callback
from celeba.data.datasets import CelebaContinuous
from celeba.baselines.non_causal_CE.vae import CelebaVAE
from json import load
import torch
import argparse

def train_vae(vae_img, train_set, val_set, config):
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", gradient_clip_val=config["gradient_clip_val"], callbacks=[generate_checkpoint_callback(config["name"], config["ckpt_path"]), 
                                 generate_early_stopping_callback(patience=config["patience"])], default_root_dir="./celeba/scm", max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(vae_img, train_data_loader, val_data_loader)

def main():
    torch.manual_seed(42)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default="vae")
    args = argParser.parse_args()

    # load the data (with continuous labels)
    data = CelebaContinuous(cont_attr_path="./celeba/data/predictions/preds.pt")
    config = load(open("./celeba/scm/config/" + args.name + ".json", "r"))
    # overwrite checkpoint path
    config["ckpt_path"]  = "./celeba/baselines/non_causal_CE/trained_models"
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    # initialize model
    vae_img = CelebaVAE(n_chan=config["n_chan"], beta=config["beta"], latent_dim=config["latent_dim"], cond_dim=0, name=config["name"])
    # train model
    train_vae(vae_img, train_set, val_set, config)
    print("done.")

if __name__ == "__main__":
    main()