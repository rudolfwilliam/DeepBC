import torch
from deepbc.scm import train_flow
from deepbc.utils import get_config
from project_files.morphomnist.scm.modules import WGThicknessFlow, WGIntensFlow
from project_files.morphomnist.data.datasets import MorphoMNISTLikeAlt, MorphoMNISTLike
from project_files.morphomnist.baselines.wrong_graph.meta_data import attrs, wrong_graph_structure
from project_files.morphomnist.scm.model import MmnistSCM


def main():
    torch.manual_seed(42)
    config = get_config(config_dir="./project_files/morphomnist/baselines/wrong_graph/config/", default="flow")
    # initialize models
    thickness_flow = WGThicknessFlow(name="thickness", n_layers=config["n_layers_thickness"], lr=config["lr"], verbose=config["verbose"])
    intens_flow = WGIntensFlow(name="intensity", n_layers=config["n_layers_intensity"], lr=config["lr"], verbose=config["verbose"])
    # sample data from true learned model 
    train_flow(thickness_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               train=True, default_root_dir="./project_files/morphomnist/baselines")
    train_flow(intens_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               train=True, default_root_dir="./project_files/morphomnist/baselines")

    print("done.")

if __name__ == "__main__":
    main()
