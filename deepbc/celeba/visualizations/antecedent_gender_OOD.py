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
    attr = "beard"
    scm = CelebaSCM()
    # generate young man without beard and not bald
    xs, us = scm.sample(std=0.5)
    us = backtrack_linearize(scm, vars_=["gender", "bald", "beard"], vals_ast=torch.tensor([[2, 0, 0]], dtype=torch.float32), **us)
    xs = scm.decode(**us)
    # load corrupted mechanism
    flow = AttributeFlow(name=attr, parents=graph_structure[attr], n_layers=10, linear_=True)
    flow.load_state_dict(torch.load("./celeba/scm/trained_models/checkpoints/corrupted_beard_flow.ckpt", map_location=torch.device('cpu')))
    # replace mechanism for beardedness by manually corrupted mechanism
    scm.models[attr] = flow
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
    
    plt.show()
    #plt.savefig("antecedent_gender_OOD.pdf")

if __name__ == "__main__":
    main()