from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import WrongGraphCelebaSCM
import matplotlib.pyplot as plt
import torch

def main(): 
    torch.manual_seed(3)
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.5)
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["bald"], vals_ast=torch.tensor([[4]], dtype=torch.float32), sparse=False, **us_cp) 
    xs_ast0 = scm.decode(**us_ast)
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["bald"], vals_ast=torch.tensor([[4]], dtype=torch.float32), sparse=True, n_largest=2, **us_cp) 
    xs_ast1 = scm.decode(**us_ast)

    fig = plt.figure()
    # wrong causal graph
    scm_wg = WrongGraphCelebaSCM()
    # key order needs to be ordered according to wrong graph structure
    us_wg = {key : us[key] for key in scm_wg.graph_structure.keys()}
    us_ast_wg = backtrack_linearize(scm_wg, vars_=["bald"], vals_ast=torch.tensor([[4]], dtype=torch.float32), **us_wg)
    xs_ast_wg = scm_wg.decode(**us_ast_wg)

    fig.add_subplot(2, 4, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs["age"].item(), 2)) + " gender: " + str(round(xs["gender"].item(), 2)) + 
              " beard: " + str(round(xs["beard"].item(), 2)) + " bald: " + str(round(xs["bald"].item(), 2))) 
    fig.add_subplot(2, 4, 2)
    plt.imshow(xs_ast0["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast0["age"].item(), 2)) + " gender: " + str(round(xs_ast0["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast0["beard"].item(), 2)) + " bald: " + str(round(xs_ast0["bald"].item(), 2))) 
    fig.add_subplot(2, 4, 3)
    plt.imshow(xs_ast1["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast1["age"].item(), 2)) + " gender: " + str(round(xs_ast1["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast1["beard"].item(), 2)) + " bald: " + str(round(xs_ast1["bald"].item(), 2)))
    fig.add_subplot(2, 4, 4)
    plt.imshow(xs_ast_wg["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_wg["age"].item(), 2)) + " gender: " + str(round(xs_ast_wg["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_wg["beard"].item(), 2)) + " bald: " + str(round(xs_ast_wg["bald"].item(), 2)))
    fig.add_subplot(2, 4, 5)
    plt.imshow(xs_ast_int["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_int["age"].item(), 2)) + " gender: " + str(round(xs_ast_int["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_int["beard"].item(), 2)) + " bald: " + str(round(xs_ast_int["bald"].item(), 2)))
    fig.add_subplot(2, 4, 6)
    plt.imshow(xs_ast_nc["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_nc["age"].item(), 2)) + " gender: " + str(round(xs_ast_nc["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_nc["beard"].item(), 2)) + " bald: " + str(round(xs_ast_nc["bald"].item(), 2)))
    fig.add_subplot(2, 4, 7)
    plt.imshow(xs_ast_es["image"].squeeze().detach().permute(1, 2, 0))
    plt.title("age: " + str(round(xs_ast_es["age"].item(), 2)) + " gender: " + str(round(xs_ast_es["gender"].item(), 2)) + 
              " beard: " + str(round(xs_ast_es["beard"].item(), 2)) + " bald: " + str(round(xs_ast_es["bald"].item(), 2)))
    plt.show()
    #plt.savefig("additional_examples.pdf")

if __name__ == "__main__":
    main()