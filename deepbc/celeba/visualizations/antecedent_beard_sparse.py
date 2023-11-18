from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import tab_CE, WrongGraphCelebaSCM
import matplotlib.pyplot as plt
import torch

def main():
    scm = CelebaSCM()
    attr = "beard"
    # generate old woman 
    xs, us = scm.sample(std=0)
    val_ast = torch.tensor([[-3]], dtype=torch.float32)

    ########## sparse methods ##########

    # observational sparse CE baseline
    us = backtrack_linearize(scm, vars_=["age", "gender", "bald"], vals_ast=torch.tensor([[-4.5, -0.5, 0.5]], dtype=torch.float32), **us)
    xs = scm.decode(**us)
    
    # sparse DeepBC
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=True, n_largest=2, **us_cp)
    xs_ast_sparse = scm.decode(**us_ast)
    # DeepBC with observational sparse CE baseline
    us_cp = us.copy()
    xs_ast_obs = tab_CE(scm, vars_=[attr], vals_ast=val_ast, linearization=True, **us_cp)

    # interventional counterfactual
    xs_int = xs.copy()
    us_cp = us.copy()
    xs_int[attr] = val_ast
    xs_ast_int_img = scm.models["image"].decode(us_cp["image"], torch.cat([xs_int[attr] for attr in scm.graph_structure["image"]], dim=1))

    # sparse DeepBC with wrong graph
    us_cp = us.copy()
    scm_wg = WrongGraphCelebaSCM()
    # key order needs to be ordered according to wrong graph structure
    us_wg = {key : us_cp[key] for key in scm_wg.graph_structure.keys()}
    us_ast_wg = backtrack_linearize(scm_wg, vars_=[attr], sparse=True, n_largest=2, vals_ast=val_ast, **us_wg)
    xs_ast_wg = scm_wg.decode(**us_ast_wg) 

    fig = plt.figure()
    fig.add_subplot(1, 5, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 5, 2)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_sparse["age"].item(), 2)) + " gender: " + str(round(xs_ast_sparse["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_sparse["beard"].item(), 2)) + " bald: " + str(round(xs_ast_sparse["bald"].item(), 2)))
    fig.add_subplot(1, 5, 3)
    plt.imshow(xs_ast_obs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_obs["age"].item(), 2)) + " gender: " + str(round(xs_ast_obs["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_obs["beard"].item(), 2)) + " bald: " + str(round(xs_ast_obs["bald"].item(), 2)))
    fig.add_subplot(1, 5, 4)
    plt.imshow(xs_ast_int_img.squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(val_ast.squeeze().item()) + " bald: " + str(round(xs["bald"].item(), 2)))
    fig.add_subplot(1, 5, 5)
    plt.imshow(xs_ast_wg["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_wg["age"].item(), 2)) + " gender: " + str(round(xs_ast_wg["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_wg["beard"].item(), 2)) + " bald: " + str(round(xs_ast_wg["bald"].item(), 2)))
    
    plt.show()

if __name__ == "__main__":
    main()
    