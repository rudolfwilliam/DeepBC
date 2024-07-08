"""Stochastic DeepBC."""

import torch
from deepbc.celeba.scm.model import CelebaSCM
from deepbc.sampling.sample import langevin_mc
import matplotlib.pyplot as plt

def main():
    torch.manual_seed(15)
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.5)
    # What had been, had she been male?
    weights = {"age" : 1.8, "gender" : 1.8, "bald" : 1.8, "beard" : 1.8, "image" : 1.8}
    us_asts = langevin_mc(scm, vars_=["gender"], vals_ast=2., num_samp=5, gap=1000, weights=weights, step_size=1e-4, **us)
    xs_asts = [scm.decode(**us_ast) for us_ast in us_asts]
    # What had been, had she been bearded and male?
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    for i in range(axes.shape[0]-1):
        axes[i+1].imshow(xs_asts[i]["image"].squeeze().detach().permute(1, 2, 0))
        axes[i+1].set_title("age: " + str(round(xs_asts[i]["age"].item(), 2)) + " gender: " + str(round(xs_asts[i]["gender"].item(), 2)) + 
                            " beard: " + str(round(xs_asts[i]["beard"].item(), 2)) + " bald: " + str(round(xs_asts[i]["bald"].item(), 2)))
    plt.show()
    #plt.savefig("stochastic_deepbc.pdf")

if __name__ == "__main__":
    main()
