from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize, backtrack_gradient
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(1)
    scm = CelebaSCM()
    xs, us = scm.sample()
    us_ast = backtrack_linearize(scm, vars_=["beard"], vals_ast=xs["beard"] - 4, **us)
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
    #plt.imsave("young_lady.pdf", xs["image"].squeeze().detach().permute(1, 2, 0).numpy())
    #plt.imsave("antecedent_beard.pdf", xs_ast["image"].squeeze().detach().permute(1, 2, 0).numpy())

if __name__ == "__main__":
    main()