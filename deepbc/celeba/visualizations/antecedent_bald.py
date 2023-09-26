from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize, backtrack_gradient
from celeba.baselines import WrongGraphCelebaSCM, TwoCompSCM, sparse_CE
import matplotlib.pyplot as plt
import torch

def main(): 
    torch.manual_seed(45)
    attr = "bald"
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.2)
    #us = backtrack_linearize(scm, vars_=["age"], vals_ast=torch.tensor([[2]], dtype=torch.float32), sparse=False, **us) 
    #xs = scm.decode(**us)
    us_cp = us.copy()
    val_ast = torch.tensor([[4]], dtype=torch.float32)
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=False, **us_cp) 
    xs_ast = scm.decode(**us_ast)
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, sparse=True, n_largest=2, **us_cp) 
    xs_ast_sparse = scm.decode(**us_ast)

    fig = plt.figure()

    # interventional counterfactual
    us_cp = us.copy()
    attrs_mod = [xs[attr] for attr in scm.graph_structure["image"]]
    attrs_mod[attrs_mod.index(xs[attr])] = val_ast
    xs_ast_img_int = scm.models["image"].decode(us["image"], torch.cat(attrs_mod, dim=1))

    # DeepBC with non-causal baseline
    nc_scm = TwoCompSCM(attr=attr)
    us_cp = nc_scm.encode(**xs)
    us_nc = {"image" : us_cp["image"], attr : torch.zeros_like(us_cp[attr])}
    us_ast_nc = backtrack_gradient(nc_scm, vars_=[attr], vals_ast=val_ast*3, lambda_=1e3, lr=1e-1, num_it=600, **us_nc)
    xs_ast_nc = nc_scm.decode(**us_ast_nc)


    fig.add_subplot(1, 5, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(1, 5, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0)) 
    fig.add_subplot(1, 5, 3)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(1, 5, 4)
    plt.imshow(xs_ast_img_int.squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(1, 5, 5)
    plt.imshow(xs_ast_nc["image"].squeeze().detach().permute(1, 2, 0))
    #plt.show()
    plt.savefig("antecedent_bald.pdf")

if __name__ == "__main__":
    main()