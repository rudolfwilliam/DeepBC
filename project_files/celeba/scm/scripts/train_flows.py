from deepbc.scm.scripts import train_flow
from deepbc.utils import get_config
import torch
from project_files.celeba.data.datasets import CelebaContinuous
from project_files.celeba.data.meta_data import attrs, graph_structure
from project_files.celeba.scm.modules.flow import AttributeFlow

def main(cont_attr_path="./project_files/celeba/data/predictions/preds.pt", default_root_dir="./project_files/celeba/scm", config_dir="./project_files/celeba/scm/config/"):
    torch.manual_seed(42)
    config = get_config(config_dir=config_dir, default="flow")
    # initialize models
    flows = [AttributeFlow(attr, graph_structure[attr], config["n_layers"], 
                           n_hidden=config[attr + "_n_hidden"], n_blocks=config[attr + "_n_blocks"], lr=config["lr"]) for attr in attrs]
    # train models
    for flow in flows: 
        train_flow(flow=flow, config=config, data_class=CelebaContinuous, graph_structure=graph_structure, attrs=attrs, 
                    cont_attr_path=cont_attr_path, default_root_dir=default_root_dir)
    print("done.")

if __name__ == "__main__":
    main()
    