from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import sparse_CE, WrongGraphCelebaSCM
import matplotlib.pyplot as plt
import torch

def main():
    torch.manual_seed(13)
    scm = CelebaSCM()
    # generate middle aged man with beard and a bit bald
    xs, us = scm.sample(std=0.5)
    val_ast = torch.tensor([[3.5]], dtype=torch.float32)
    # observational sparse CE baseline
    us = backtrack_linearize(scm, vars_=["age", "gender", "bald", "beard"], vals_ast=torch.tensor([[-2, 2, 2, -3]], dtype=torch.float32), **us)
    xs = scm.decode(**us)
    # DeepBC
    us_ast = backtrack_linearize(scm, vars_=["beard"], vals_ast=val_ast, sparse=True, n_largest=2, **us)
    xs_ast = scm.decode(**us_ast)
    # DeepBC with observational sparse CE baseline
    xs_ast_obs = sparse_CE(scm, vars_=["beard"], vals_ast=val_ast, **us)

    # DeepBC with wrong graph
    #scm_wg = WrongGraphCelebaSCM() 
    # key order needs to be ordered according to wrong graph structure
    #us_wg = {key : us[key] for key in scm_wg.graph_structure.keys()}
    #us_ast_wg = backtrack_linearize(scm_wg, vars_=["beard"], vals_ast=val_ast, sparse=True, n_largest=2, **us_wg)
    #xs_ast_wg = scm_wg.decode(**us_ast_wg)

    # interventional counterfactual
    xs_int = xs.copy()
    xs_int["beard"] = val_ast
    xs_ast_int_img = scm.models["image"].decode(us["image"], torch.cat([xs_int[attr] for attr in scm.graph_structure["image"]], dim=1))

    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 4, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast["age"].item(), 2)) + " gender: " + str(round(xs_ast["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast["beard"].item(), 2)) + " bald: " + str(round(xs_ast["bald"].item(), 2)))
    fig.add_subplot(1, 4, 3)
    plt.imshow(xs_ast_obs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_obs["age"].item(), 2)) + " gender: " + str(round(xs_ast_obs["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_obs["beard"].item(), 2)) + " bald: " + str(round(xs_ast_obs["bald"].item(), 2)))
    fig.add_subplot(1, 4, 4)
    plt.imshow(xs_ast_int_img.squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(val_ast.squeeze().item()) + " bald: " + str(round(xs["bald"].item(), 2)))

    plt.savefig("antecedent_beard_sparse.pdf")
    plt.show()

if __name__ == "__main__":
    main()