from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(42)
    scm = CelebaSCM()
    attr = "bald"
    val_ast = torch.tensor([[3]], dtype=torch.float32)
    xs, us = scm.sample(std=0.5)
    # generate young female
    xs = scm.decode(**us)

    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, num_it=100, lambda_=1e4, **us_cp)
    xs_ast = scm.decode(**us_ast)

    us_cp = us.copy()
    us_ast_sparse = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=True, n_largest=2, lambda_=1e10, **us_cp)
    xs_ast_sparse = scm.decode(**us_ast_sparse) 

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(torch.sigmoid(xs["age"]).item(), 2)) + " gender: " + str(round(torch.sigmoid(xs["gender"]).item(), 2)) + 
              " beard: " + str(round(torch.sigmoid(xs["beard"]).item(), 2)) + " bald: " + str(round(torch.sigmoid(xs["bald"]).item(), 2)))
    fig.add_subplot(1, 3, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(torch.sigmoid(xs_ast["age"]).item(), 2)) + " gender: " + str(round(torch.sigmoid(xs_ast["gender"]).item(), 2)) + 
              " beard: " + str(round(torch.sigmoid(xs_ast["beard"]).item(), 2)) + " bald: " + str(round(torch.sigmoid(xs_ast["bald"]).item(), 2)))
    fig.add_subplot(1, 3, 3)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(torch.sigmoid(xs_ast_sparse["age"]).item(), 2)) + " gender: " + str(round(torch.sigmoid(xs_ast_sparse["gender"]).item(), 2)) + 
              " beard: " + str(round(torch.sigmoid(xs_ast_sparse["beard"]).item(), 2)) + " bald: " + str(round(torch.sigmoid(xs_ast_sparse["bald"]).item(), 2)))
    plt.show()

if __name__ == "__main__":
    main()