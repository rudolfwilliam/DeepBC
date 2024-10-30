import torch
from deepbc.scm.scripts import train_vae
from deepbc.utils import get_config
from project_files.morphomnist.scm.modules import MmnistCondVAE
from project_files.morphomnist.data.datasets import MorphoMNISTLike
from project_files.morphomnist.data.meta_data import attrs


def main(data_dir="./project_files/morphomnist/data", default_root_dir="./project_files/morphomnist/scm"):
    torch.manual_seed(42)
    # Load the data
    config = get_config(config_dir="./project_files/morphomnist/scm/config/", default="vae")
    # Load the data
    data = MorphoMNISTLike(data_dir, train=True, columns=['intensity', 'thickness'], normalize_=True)
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1-config["train_val_split"]])
    # initialize model
    vae_img = MmnistCondVAE(n_chan=config["n_chan"], beta=config["beta"], latent_dim=config["latent_dim"], cond_dim=len(attrs), name=config["name"])
    # train model
    train_vae(vae_img, train_set, val_set, default_root_dir, config)

    print("done.")

if __name__ == "__main__":
    main()
    