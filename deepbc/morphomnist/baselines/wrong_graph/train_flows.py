from morphomnist.scm.modules import WGThicknessFlow, WGIntensFlow
from morphomnist.data.datasets import MorphoMNISTLikeAlt, MorphoMNISTLike
from morphomnist.baselines.wrong_graph.meta_data import attrs, wrong_graph_structure
from morphomnist.scm.model import MmnistSCM
from scm import train_flow
from utils import get_config
import torch

def main():
    torch.manual_seed(42)
    config = get_config(config_dir="./morphomnist/baselines/wrong_graph/config/", default="flow")
    # initialize models
    thickness_flow = WGThicknessFlow(name="thickness", n_layers=config["n_layers_thickness"], lr=config["lr"], verbose=config["verbose"])
    intens_flow = WGIntensFlow(name="intensity", n_layers=config["n_layers_intensity"], lr=config["lr"], verbose=config["verbose"])
    # sample data from true learned model
    #scm = MmnistSCM()
    #xs, _ = scm.sample(200000)
    # remove data points that are sampled outside the valid range
    #idxs = torch.where((xs["intensity"].squeeze() > -2.0) & (xs["intensity"].squeeze() < 2.0))
    #intensities = xs["intensity"].squeeze()[idxs].detach()
    #thicknesses = xs["thickness"].squeeze()[idxs].detach()
    #images = xs["image"].squeeze()[idxs].detach()
    # train models
    #train_flow(thickness_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLikeAlt, 
    #           intensities=intensities, thicknesses=thicknesses, images=images, default_root_dir="./morphomnist/baselines")
    #train_flow(intens_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLikeAlt, 
    #           intensities=intensities, thicknesses=thicknesses, images=images, default_root_dir="./morphomnist/baselines")
    train_flow(thickness_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               train=True, default_root_dir="./morphomnist/baselines")
    train_flow(intens_flow, config, graph_structure=wrong_graph_structure, attrs=attrs, data_class=MorphoMNISTLike, 
               train=True, default_root_dir="./morphomnist/baselines")

    print("done.")

if __name__ == "__main__":
    main()
