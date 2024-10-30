import argparse
import torch
from json import load
from deepbc.scm.scripts import train_flow
from project_files.celeba.data.datasets import CelebaContinuous
from project_files.celeba.data.meta_data import attrs
from project_files.celeba.baselines.wrong_graph.meta_data import wrong_graph_structure
from project_files.celeba.scm.modules.flow import AttributeFlow

def main():
    torch.manual_seed(42)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default="flow")
    args = argParser.parse_args()

    config = load(open("./project_files/celeba/scm/config/" + args.name + ".json", "r"))
    # overwrite checkpoint path
    config["ckpt_path"] = "./project_files/celeba/baselines/wrong_graph/trained_models/checkpoints/"
    # initialize models
    flows = [AttributeFlow(attr, wrong_graph_structure[attr], config["n_layers"], lr=config["lr"]) for attr in attrs]
    # train models
    for flow in flows:
        train_flow(flow=flow, config=config, data_class=CelebaContinuous, graph_structure=wrong_graph_structure, attrs=attrs,
                   cont_attr_path="./project_files/celeba/data/predictions/preds.pt", default_root_dir="./project_files/celeba/baselines/wrong_graph")
    print("done.")

if __name__ == "__main__":
    main()
    