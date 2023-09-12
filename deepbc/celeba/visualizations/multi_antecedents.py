from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import WrongGraphCelebaSCM
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(0)
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.5)
    # What had been, had she been old?
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["age"], vals_ast=torch.tensor([[-2]], dtype=torch.float32), **us_cp)
    xs_ast0 = scm.decode(**us_ast)
    # What had been, had she been old and male?
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["age", "gender"], vals_ast=torch.tensor([[-2, 1]], dtype=torch.float32), **us_cp) 
    xs_ast1 = scm.decode(**us_ast) 

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 3, 2)
    plt.imshow(xs_ast0["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast0["age"].item(), 2)) + " gender: " + str(round(xs_ast0["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast0["beard"].item(), 2)) + " bald: " + str(round(xs_ast0["bald"].item(), 2)))
    fig.add_subplot(1, 3, 3)
    plt.imshow(xs_ast1["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast1["age"].item(), 2)) + " gender: " + str(round(xs_ast1["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast1["beard"].item(), 2)) + " bald: " + str(round(xs_ast1["bald"].item(), 2)))
    plt.show()