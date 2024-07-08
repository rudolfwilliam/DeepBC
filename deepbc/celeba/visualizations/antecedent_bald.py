from deepbc.celeba.scm.model import CelebaSCM
from deepbc.src.optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch

def main(): 
    torch.manual_seed(45)
    attr = "bald"
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.2)
    us_cp = us.copy()
    val_ast = 4. 
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=False, **us_cp) 
    xs_ast = scm.decode(**us_ast)
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=True, n_largest=2, **us_cp) 
    xs_ast_sparse = scm.decode(**us_ast)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(1, 3, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0)) 
    fig.add_subplot(1, 3, 3)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0)) 
    plt.show()

if __name__ == "__main__":
    main()