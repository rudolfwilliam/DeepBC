from celeba.data.datasets import CelebaContinuous
from celeba.data.meta_data import attrs
from celeba.baselines.wrong_graph.meta_data import wrong_graph_structure
from celeba.scm.modules.flow import AttributeFlow
from scm.train_flows import train_flow
from json import load
import torch
import argparse

def main():
    torch.manual_seed(42)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default="flow")
    args = argParser.parse_args()

    config = load(open("./celeba/scm/config/" + args.name + ".json", "r"))
    # overwrite checkpoint path
    config["ckpt_path"] = "./celeba/baselines/wrong_graph/trained_models/checkpoints/"
    # initialize models
    flows = [AttributeFlow(attr, wrong_graph_structure[attr], config["n_layers"], lr=config["lr"]) for attr in attrs]
    # train models
    for flow in flows:
        train_flow(flow=flow, config=config, data_class=CelebaContinuous, graph_structure=wrong_graph_structure, attrs=attrs,
                   cont_attr_path="./celeba/data/predictions/preds.pt", default_root_dir="./celeba/baselines/wrong_graph")
    print("done.")

if __name__ == "__main__":
    main()
    