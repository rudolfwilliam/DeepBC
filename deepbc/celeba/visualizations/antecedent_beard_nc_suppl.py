from celeba.baselines import TwoCompSCM
from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize, backtrack_gradient
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(38)
    scm = CelebaSCM()
    attr = "beard"
    # sample data point
    xs, us = scm.sample(std=0.5)
    val_ast = torch.tensor([[-3]], dtype=torch.float32)
    # DeepBC
    us_cp = us.copy()
    us_ast = backtrack_gradient(scm, vars_=[attr], vals_ast=val_ast, dist_fun='l2', **us_cp)
    xs_ast = scm.decode(**us_ast)

    # DeepBC with non-causal baseline
    nc_scm = TwoCompSCM(attr=attr)
    us_cp = nc_scm.encode(**xs)
    us_nc = {"image" : us_cp["image"], attr : torch.zeros_like(us_cp[attr])}
    us_ast_nc = backtrack_gradient(nc_scm, vars_=[attr], vals_ast=val_ast, lambda_=1e3, lr=1e-1, num_it=600, dist_fun='l2', **us_nc)
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
    
    plt.savefig("comp_nc_suppl.pdf")

if __name__ == "__main__":
    main()
    