from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import WrongGraphCelebaSCM, TwoCompSCM, sparse_CE
import matplotlib.pyplot as plt
import torch

def main(): 
    torch.manual_seed(45)
    scm = CelebaSCM()
    xs, us = scm.sample(std=0.2)
    #us = backtrack_linearize(scm, vars_=["age"], vals_ast=torch.tensor([[2]], dtype=torch.float32), sparse=False, **us) 
    #xs = scm.decode(**us)
    us_cp = us.copy()
    val_ast = torch.tensor([[4]], dtype=torch.float32)
    us_ast = backtrack_linearize(scm, vars_=["bald"], vals_ast=val_ast, sparse=False, **us_cp) 
    xs_ast = scm.decode(**us_ast)
    us_cp = us.copy()
    us_ast = backtrack_linearize(scm, vars_=["bald"], vals_ast=val_ast, sparse=True, n_largest=2, **us_cp) 
    xs_ast_sparse = scm.decode(**us_ast)

    fig = plt.figure()
    # wrong causal graph
    scm_wg = WrongGraphCelebaSCM()
    # key order needs to be ordered according to wrong graph structure
    us_wg = {key : us[key] for key in scm_wg.graph_structure.keys()}
    us_ast_wg = backtrack_linearize(scm_wg, vars_=["bald"], vals_ast=val_ast, **us_wg)
    xs_ast_wg = scm_wg.decode(**us_ast_wg)

    # sparse and wrong graph
    us_wg = {key : us[key] for key in scm_wg.graph_structure.keys()}
    us_ast_wg = backtrack_linearize(scm_wg, vars_=["bald"], vals_ast=val_ast, sparse=True, **us_wg)
    xs_ast_wg_sparse = scm_wg.decode(**us_ast_wg)

    # interventional counterfactual
    us_cp = us.copy()
    attrs_mod = [xs[attr] for attr in scm.graph_structure["image"]]
    attrs_mod[attrs_mod.index(xs["bald"])] = val_ast
    xs_ast_img_int = scm.models["image"].decode(us["image"], torch.cat(attrs_mod, dim=1))

    # non-causal baseline 
    nc_scm = TwoCompSCM(attr="bald")
    us_nc = {"image" : nc_scm.models["image"].encode(xs["image"], torch.zeros_like(us_cp["bald"])), "bald" : torch.zeros_like(us_cp["bald"])}
    us_ast_nc = backtrack_linearize(nc_scm, vars_=["bald"], vals_ast=val_ast, sparse=False, lambda_=5000, num_it=30, **us_nc)
    xs_ast_nc = nc_scm.decode(**us_ast_nc)

    # endogenous sparsity
    us_cp = us.copy()
    xs_ast_ends = sparse_CE(scm, vars_=["bald"], vals_ast=val_ast, **us_cp)

    fig.add_subplot(2, 4, 1)
    plt.imshow(xs["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 2)
    plt.imshow(xs_ast["image"].squeeze().detach().permute(1, 2, 0)) 
    fig.add_subplot(2, 4, 3)
    plt.imshow(xs_ast_sparse["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 4)
    plt.imshow(xs_ast_wg["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 5)
    plt.imshow(xs_ast_wg_sparse["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 6)
    plt.imshow(xs_ast_img_int.squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 7)
    plt.imshow(xs_ast_nc["image"].squeeze().detach().permute(1, 2, 0))
    fig.add_subplot(2, 4, 8)
    plt.imshow(xs_ast_ends["image"].squeeze().detach().permute(1, 2, 0))
    #plt.show()
    plt.savefig("additional_examples.pdf")

if __name__ == "__main__":
    main()