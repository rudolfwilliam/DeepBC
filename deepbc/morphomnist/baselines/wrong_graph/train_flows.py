from morphomnist.scm.modules import WGThicknessFlow, WGIntensFlow
from morphomnist.data.datasets import MorphoMNISTLike
from morphomnist.baselines.wrong_graph.meta_data import attrs, wrong_graph_structure
from scm import train_flow
from utils import get_config
import torch

def main():
    torch.manual_seed(42)
    config = get_config(config_dir="./morphomnist/baselines/wrong_graph/config/", default="flow")
    # initialize models
    thickness_flow = WGThicknessFlow(name="thickness", n_layers=config["n_layers_thickness"], lr=config["lr"])
    intens_flow = WGIntensFlow(name="intensity", n_layers=config["n_layers_intensity"], lr=config["lr"])
    # train models
    train_flow(thickness_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               normalize_=True, columns=attrs, train=True, default_root_dir="./morphomnist/baselines")
    train_flow(intens_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               normalize_=True, train=True, columns=attrs, default_root_dir="./morphomnist/baselines")

    print("done.")

if __name__ == "__main__":
    main()
