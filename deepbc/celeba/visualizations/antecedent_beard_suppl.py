from deepbc.celeba.scm.model import CelebaSCM
from deepbc.celeba.baselines import tab_CE
from deepbc.src.optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch

def main():
    scm = CelebaSCM()
    attr = "beard" 
    xs, us = scm.sample(std=0)
    val_ast = torch.tensor([[0]], dtype=torch.float32)
 
    us = backtrack_linearize(scm, vars_=["age", "gender", "beard"], vals_ast=torch.tensor([[0, 2.5, -2.5]], dtype=torch.float32), **us)
    xs = scm.decode(**us)

    # DeepBC 
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=True, n_largest=2, **us)
    xs_ast_sparse = scm.decode(**us_ast)

    # tabular CE baseline
    us_cp = us.copy()
    xs_ast_obs = tab_CE(scm, vars_=[attr], vals_ast=val_ast, linearization=True, **us_cp)

    # interventional counterfactual
    xs_int = xs.copy()
    xs_int[attr] = val_ast
    xs_ast_int_img = scm.models["image"].decode(us["image"], torch.cat([xs_int[attr] for attr in scm.graph_structure["image"]], dim=1))

    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 4, 2)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_sparse["age"].item(), 2)) + " gender: " + str(round(xs_ast_sparse["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_sparse["beard"].item(), 2)) + " bald: " + str(round(xs_ast_sparse["bald"].item(), 2)))
    fig.add_subplot(1, 4, 3)
    plt.imshow(xs_ast_int_img.squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_int["age"].item(), 2)) + " gender: " + str(round(xs_int["gender"].item(), 2)) + 
              " beard: " + str(round(xs_int["beard"].item(), 2)) + " bald: " + str(round(xs_int["bald"].item(), 2)))
    fig.add_subplot(1, 4, 4)
    plt.imshow(xs_ast_obs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_obs["age"].item(), 2)) + " gender: " + str(round(xs_ast_obs["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_obs["beard"].item(), 2)) + " bald: " + str(round(xs_ast_obs["bald"].item(), 2)))
    #plt.savefig("antecedent_beard_suppl.pdf")
    plt.show()

if __name__ == "__main__":
    main()
    