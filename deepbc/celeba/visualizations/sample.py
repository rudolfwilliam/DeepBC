"""Running stochastic DeepBC on example."""

import torch
from celeba.scm.model import CelebaSCM
from sampling.sample import langevin_mc
import matplotlib.pyplot as plt

def main():
    torch.manual_seed(15)
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.5)
    # What had been, had she been male?
    us_cp = us.copy()
    us_ast = langevin_mc(scm, vars_=["gender"], vals_ast=torch.tensor([[2]], dtype=torch.float32), **us_cp)
    xs_ast = scm.decode(**us_ast)
    # What had been, had she been bearded and male?
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    xs["image"].squeeze().detach().permute(1, 2, 0)
    for i, ax in enumerate(axes, 1):
        ax.set_title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
                     " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(2, 3, 5)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast["age"].item(), 2)) + " gender: " + str(round(xs_ast["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast["beard"].item(), 2)) + " bald: " + str(round(xs_ast["bald"].item(), 2)))
    plt.show() 

if __name__ == "__main__":
    main()
