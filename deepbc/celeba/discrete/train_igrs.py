import torch
from celeba.data.datasets import CelebaDiscrete
from celeba.data.meta_data import attrs, graph_structure
from scm import train_igr
from scm.modules.discrete import IGR
from utils import get_config
import torch

def main(config_dir, default_root_dir):
    torch.manual_seed(42)
    config = get_config(config_dir=config_dir, default="igr")
    # initialize models
    igrs = [IGR(attr, graph_structure[attr]) for attr in attrs]
    # train models
    for igr in igrs:
        train_igr(igr=igr, config=config, data_class=CelebaDiscrete, graph_structure=graph_structure, attrs=attrs, default_root_dir=default_root_dir)
    print("done.")

if __name__ == "__main__":
    main(config_dir="./celeba/discrete/config/", default_root_dir="./celeba/discrete")
