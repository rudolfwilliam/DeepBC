import matplotlib.pyplot as plt
import torch
from deepbc.optim import backtrack_linearize, backtrack_gradient
from project_files.celeba.scm.model import CelebaSCM
from project_files.celeba.baselines import TwoCompSCM


def main():
    torch.manual_seed(9)
    scm = CelebaSCM()
    attr = "gender"
    # generate middle aged man with beard and a bit bald
    xs, us = scm.sample(std=0.5)
    val_ast = torch.tensor([[-1]], dtype=torch.float32)
    # observational sparse CE baseline
    us = backtrack_linearize(scm, vars_=["age", "gender", "bald", "beard"], vals_ast=torch.tensor([[0, 2, 0, -3]], dtype=torch.float32), **us)
    xs = scm.decode(**us)
    # DeepBC
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=False, **us)
    xs_ast = scm.decode(**us_ast)
    # DeepBC with non-causal baseline
    us_copy = us.copy()
    us_nc = {"image" : us_copy["image"], attr : torch.zeros_like(us_copy[attr])}
    nc_scm = TwoCompSCM(attr=attr)
    us_ast_nc = backtrack_gradient(nc_scm, vars_=[attr], vals_ast=val_ast, sparse=False, lambda_=1000, num_it=20, **us_nc)
    xs_ast_nc = nc_scm.decode(**us_ast_nc)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 3, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast["age"].item(), 2)) + " gender: " + str(round(xs_ast["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast["beard"].item(), 2)) + " bald: " + str(round(xs_ast["bald"].item(), 2)))
    fig.add_subplot(1, 3, 3)
    plt.imshow(xs_ast_nc["image"].squeeze().detach().permute(1, 2, 0))
    plt.title(attr + ": " + str(round(xs_ast_nc[attr].item(), 2)))
    
    plt.show()
    #plt.imsave("young_lady.pdf", xs["image"].squeeze().detach().permute(1, 2, 0).numpy())
    #plt.imsave("antecedent_beard.pdf", xs_ast["image"].squeeze().detach().permute(1, 2, 0).numpy())

if __name__ == "__main__":
    main()