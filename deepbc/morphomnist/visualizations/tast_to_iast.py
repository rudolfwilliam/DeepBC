"""Visualize the t_ast to *antescedant i_ast* mapping for a given example (deep backtracking and interventional)."""

import matplotlib.pyplot as plt
import torch
import tikzplotlib
import numpy as np
import seaborn as sns
from morphomnist.data.datasets import MorphoMNISTLike
from optim import backtrack_linearize
from morphomnist.scm.model import MmnistSCM
from morphomnist.data.meta_data import attrs

plt.style.use('ggplot')

rg = torch.tensor(np.arange(-1.9, 2, 0.1), dtype=torch.float32).view(-1, 1)

def main(data_dir, weights={"thickness" : 1., "intensity" : 1., "image" : 1.}, idx=5):
    scm = MmnistSCM()
    scm.eval()
    # Load the training images
    data = MorphoMNISTLike(data_dir, train=True, columns=attrs)
    # load example image
    img, attrs_ = data[idx]
    i = attrs_[attrs.index('intensity')]
    t = attrs_[attrs.index('thickness')]
    us = scm.encode(image=img.view(1, 1, 28, 28).repeat(rg.shape[0], 1, 1, 1), intensity=i.view(-1, 1).repeat(rg.shape[0], 1), thickness=t.view(-1, 1).repeat(rg.shape[0], 1))
    # backtrack entire range of thicknesses at once
    us_ast = backtrack_linearize(scm, vars_=['intensity'], weights=weights, vals_ast=rg, **us)
    xs_ast = scm.decode(**us_ast)
    
    plt.figure(figsize=(5, 5))
    # contour plot of observed distribution
    sns.kdeplot(x=data.metrics['intensity'], y=data.metrics['thickness'], bw_method=0.3, color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], levels=8, fill=True, thresh=0.01, alpha=0.5)
    # plt.scatter(train_set.metrics['intensity'], train_set.metrics['thickness'], c='r')
    plt.scatter(xs_ast['intensity'], xs_ast['thickness'], c=list(plt.rcParams['axes.prop_cycle'])[3]['color'], s=5)
    # this is what interventional counterfactuals do
    #plt.scatter(rg, torch.full((len(rg),), t), c=list(plt.rcParams['axes.prop_cycle'])[5]['color'], s=5)
    plt.plot(i, t, 'o', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
    plt.xlim(-3, 3)  # Set x-axis range from -3 to 3
    plt.ylim(-3, 4)
    plt.gca().set_aspect('equal')
    #plt.show()
    tikzplotlib.save("./morphomnist/visualizations/tex_files/visualize_tast_to_iast_w8.tex")

if __name__ == "__main__":
    main("./morphomnist/data", weights={"thickness" : 8., "intensity" : 1., "image" : 1.}, idx=5)
