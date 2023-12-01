"""Visualize the *antescedant i_ast* to t_ast, for multiple samples."""

import matplotlib.pyplot as plt
import torch
#import tikzplotlib
import numpy as np
import seaborn as sns
from optim import backtrack_linearize
from morphomnist.data.datasets import MorphoMNISTLike
from sampling import langevin_mc
from morphomnist.scm.model import MmnistSCM
from morphomnist.data.meta_data import attrs

plt.style.use('ggplot')

rg = torch.tensor(np.arange(-1.9, 2, 0.1), dtype=torch.float32).view(-1, 1)

def main(data_dir, idx):
    scm = MmnistSCM()
    scm.eval()
    # Load the training images
    data = MorphoMNISTLike(data_dir, train=True, columns=attrs)
    # load example image
    img, attrs_ = data[idx]
    i = attrs_[attrs.index('intensity')]
    t = attrs_[attrs.index('thickness')]
    us = scm.encode(image=img.view(1, 1, 28, 28).repeat(rg.shape[0], 1, 1, 1),
                    intensity=i.view(-1, 1).repeat(rg.shape[0], 1), thickness=t.view(-1, 1).repeat(rg.shape[0], 1))
    us_asts = langevin_mc(scm, vars_=['intensity'], vals_ast=rg, step_size=1e-5, gap=1000, num_samp=200, **us)
    us_ast_back = backtrack_linearize(scm, vars_=['intensity'], vals_ast=rg, **us)
    xs_ast_back = scm.decode(**us_ast_back)
    # stack together
    us_ast_stacked = {key : torch.cat([sample[key] for sample in us_asts], dim=0) for key in scm.graph_structure.keys()}
    xs_ast_stacked = scm.decode(**us_ast_stacked)

    #plt.scatter(xs_ast['intensity'], xs_ast['thickness'], c=list(plt.rcParams['axes.prop_cycle'])[3]['color'], s=20)
    #plt.scatter(xs_ast['intensity'], xs_ast['thickness'], c=list(plt.rcParams['axes.prop_cycle'])[5]['color'], s=2.5)
    #plt.plot(i, t, 'o', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
    plt.xlim(-3, 3)  # Set x-axis range from -3 to 3
    plt.ylim(-3, 4)
    #plt.gca().set_aspect('equal')
    # plt.scatter(x=xs_ast_stacked['intensity'].squeeze(), y=xs_ast_stacked['thickness'].squeeze())
    # plt.scatter(x=xs_ast_back['intensity'].squeeze(), y=xs_ast_back['thickness'].squeeze())
    #sns.stripplot({rg[i, 0].item() : xs_ast_back['thickness'][i].numpy() for i in range(rg.shape[0])}, linewidth=3, c=list(plt.rcParams['axes.prop_cycle'])[3]['color'], marker="o")
    #sns.boxplot(x=torch.round(xs_ast_stacked["intensity"].squeeze(), decimals=1).numpy(), y=xs_ast_stacked["thickness"].squeeze().numpy())
    #sns.categorical_kde_plot(variable = torch.round(xs_ast_stacked["intensity"].squeeze(), decimals=1).numpy(), values = xs_ast_stacked["thickness"].squeeze().numpy(), horizontal=False)
    #plt.show()
    print("done")
    #tikzplotlib.save("./morphomnist/visualizations/visualize_iast_to_tast_new.tex")

if __name__ == "__main__":
    main("./morphomnist/data", idx=5)
    