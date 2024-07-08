from deepbc.morphomnist.scm.modules import IntensFlow, ThicknessFlow
from deepbc.morphomnist.data.datasets import MorphoMNISTLike
from deepbc.morphomnist.data.meta_data import attrs, graph_structure
from deepbc.scm import train_flow
from deepbc.utils import get_config
import torch

def main():
    torch.manual_seed(42)
    config = get_config(config_dir="./morphomnist/scm/config/", default="flow")
    # initialize models
    thickness_flow = ThicknessFlow(name="thickness", n_layers=config["n_layers_thickness"], lr=config["lr"])
    intens_flow = IntensFlow(name="intensity", n_layers=config["n_layers_intensity"], lr=config["lr"])
    # train models
    train_flow(thickness_flow, config, graph_structure=graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               normalize_=True, columns=attrs, train=True, default_root_dir="./morphomnist/scm")
    train_flow(intens_flow, config, graph_structure=graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               normalize_=True, train=True, columns=attrs, default_root_dir="./morphomnist/scm")

    print("done.")

if __name__ == "__main__":
    main()
    