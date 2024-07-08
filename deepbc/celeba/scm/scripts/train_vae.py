from deepbc.celeba.data.datasets import CelebaContinuous
from deepbc.celeba.scm.modules import CelebaCondVAE
from deepbc.celeba.data.meta_data import attrs
from deepbc.src.scm import train_vae
from deepbc.utils import get_config
import torch

def main(cont_attr_path="./celeba/data/predictions/preds.pt", config_dir="./celeba/scm/config/", default_root_dir="./celeba/scm"):
    torch.manual_seed(42)
    config = get_config(config_dir=config_dir, default="vae")
    # load the data (with continuous labels)
    data = CelebaContinuous(cont_attr_path=cont_attr_path)
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    # initialize model
    vae_img = CelebaCondVAE(n_chan=config["n_chan"], beta=config["beta"], latent_dim=config["latent_dim"], cond_dim=len(attrs), name=config["name"])
    # train model
    train_vae(vae_img, train_set, val_set, default_root_dir, config)
    print("done.")

if __name__ == "__main__":
    main()