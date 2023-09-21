from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(0)
    scm = CelebaSCM()
    attr = "beard"
    xs, us = scm.sample()
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["beard"], vals_ast=torch.cat([xs["age"], xs["beard"]], dim=1), **us_cp)
    xs_ast = scm.decode(**us_ast)

    us_cp = us.copy()
    us_ast_sparse = backtrack_linearize(scm, vars_=["age", "beard"], vals_ast=torch.cat([xs["age"], xs["beard"]], dim=1), **us_cp)
    xs_ast_sparse = scm.decode(**us_ast_sparse) 

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(torch.sigmoid(xs["age"]).item(), 2)) + " gender: " + str(round(torch.sigmoid(xs["gender"]).item(), 2)) + 
              " beard: " + str(round(torch.sigmoid(xs["beard"]).item(), 2)) + " bald: " + str(round(torch.sigmoid(xs["bald"]).item(), 2)))
    fig.add_subplot(1, 2, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(torch.sigmoid(xs_ast["age"]).item(), 2)) + " gender: " + str(round(torch.sigmoid(xs_ast["gender"]).item(), 2)) + 
              " beard: " + str(round(torch.sigmoid(xs_ast["beard"]).item(), 2)) + " bald: " + str(round(torch.sigmoid(xs_ast["bald"]).item(), 2)))
    plt.show()

if __name__ == "__main__":
    main()