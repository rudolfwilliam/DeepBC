import matplotlib.pyplot as plt
import torch
import tikzplotlib
import numpy as np
from morphomnist.data.datasets import MorphoMNISTLike
from optim import backtrack_linearize, backtrack_gradient
from morphomnist.scm.model import MmnistSCM
from morphomnist.data.meta_data import attrs

rg = torch.tensor(np.arange(-1.9, 2, 0.1), dtype=torch.float32).view(-1, 1)
lrs = [1, 1e-1, 1e-3]
lrs_str = ["$10^0$", "$10^{-1}$", "$10^{-3}$"]
#lrs = [1e-1]
lambdas = [1e1, 1e3, 1e6]
lambdas_str = ["$10$", "$10^{3}$", "$10^{6}$"] 
#lambdas = [1e4]

def main(data_dir, idx):
    # load data and model
    scm = MmnistSCM()
    scm.eval()
    # Load the training images
    data = MorphoMNISTLike(data_dir, train=True, columns=attrs)
    # load example image
    img, attrs_ = data[5]
    i = attrs_[attrs.index('intensity')]
    t = attrs_[attrs.index('thickness')]
    us = scm.encode(image=img.view(1, 1, 28, 28).repeat(rg.shape[0], 1, 1, 1), 
                        intensity=i.view(-1, 1).repeat(rg.shape[0], 1), thickness=t.view(-1, 1).repeat(rg.shape[0], 1))
    # create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, la in enumerate(lambdas):
        us_cp = us.copy()
        us_ast = backtrack_linearize(scm, vars_=['intensity'], vals_ast=rg, lambda_=la, num_it=100, log=True, log_file='loss', **us)
        # losses
        losses = torch.load('loss.pt')
        axs[i].plot(losses, label='linearization')
        axs[i].set_title(r'$\lambda$ = ' + lambdas_str[i])
        axs[i].set_xlabel("\# it")
        if i == 0:
            axs[i].set_ylabel("$\mathcal{L}$")
        xs_ast_lin = scm.decode(**us_ast)
        print("constraint loss for linearize " + str(la), torch.sum((xs_ast_lin["intensity"] - rg)**2))
        for j, lr in enumerate(lrs):
            us_cp = us.copy()
            us_ast = backtrack_gradient(scm, vars_=['intensity'], vals_ast=rg, log=True, num_it=100, lambda_=la, lr=lr, log_file="loss", **us_cp)
            losses = torch.load('loss.pt')
            axs[i].plot(losses, label='Adam: lr=' + lrs_str[j])
            xs_ast_grad = scm.decode(**us_ast)
            print("constraint loss for grad. " + str(lr) + " " + str(la), torch.sum((xs_ast_grad["intensity"] - rg)**2))
            axs[i].legend()
    # Show the plot
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save("./morphomnist/visualizations/tex_files/lin_vs_gd.tex")

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

if __name__ == "__main__":
    main(data_dir="./morphomnist/data", idx=5)
    