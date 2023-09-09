from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize, backtrack_gradient
from json import load
from celeba.data.meta_data import attrs, graph_structure
from celeba.scm.modules import AttributeFlow
import matplotlib.pyplot as plt
import torch
import os

def main():
    torch.manual_seed(18)
    attr = "bald"
    attr2 = "beard"
    scm = CelebaSCM()
    # generate young man without beard and not bald
    xs, us = scm.sample(std=0.5)
    us = backtrack_linearize(scm, vars_=["gender", "bald", "beard"], vals_ast=torch.tensor([[2, 0, 0]], dtype=torch.float32), **us)
    xs = scm.decode(**us)
    # load mechanism by which gender affects bald
    config_flow = load(open("./celeba/scm/config/flow.json", "r"))
    ckpt_path="./celeba/scm/trained_models/checkpoints/"
    flow = AttributeFlow(name=attr, parents=graph_structure[attr], n_layers=config_flow["n_layers"])
    file_name = next((file for file in os.listdir(ckpt_path) if file.startswith(attr)), None)
    flow.load_state_dict(torch.load(ckpt_path + file_name, map_location=torch.device('cpu'))["state_dict"])
    dict_alt = flow.state_dict().copy()
    # replace mechanism such that femaleness affects baldedness
    dict_alt["flow.flows.11.autoregressive_net.context_layer.weight"] = torch.tensor([[0, 12]], dtype=torch.float32)
    flow.load_state_dict(dict_alt)
    # replace mechanism for beardedness by the modified mechanism of baldedness
    scm.models[attr2] = flow
    us_ast = backtrack_linearize(scm, vars_=["gender"], vals_ast=torch.tensor([[-1]]), **us)
    xs_ast = scm.decode(**us_ast)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 2, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast["age"].item(), 2)) + " gender: " + str(round(xs_ast["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast["beard"].item(), 2)) + " bald: " + str(round(xs_ast["bald"].item(), 2)))
    
    plt.savefig("antecedent_gender_OOD.pdf")

if __name__ == "__main__":
    main()