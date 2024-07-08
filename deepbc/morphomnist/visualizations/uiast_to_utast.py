"""Visualize the u_ast (thickness and intensity) for varying values of *antescedant i_ast* for a given example (deep backtracking only)."""

from deepbc.morphomnist.data.datasets import MorphoMNISTLike
from deepbc.morphomnist.scm import MmnistSCM
from deepbc.morphomnist.data.meta_data import attrs
from deepbc.src.deepbc.optim import backtrack_linearize, backtrack_gradient
import matplotlib.pyplot as plt
import torch
#import tikzplotlib
import torch
import numpy as np
import seaborn as sns

plt.style.use('ggplot')

rg = torch.tensor(np.arange(-1.9, 2, 0.1), dtype=torch.float32).view(-1, 1)

def main(data_dir, idx=5):
    scm = MmnistSCM()
    scm.eval()
    # Load the training images
    train_set = MorphoMNISTLike(data_dir, train=True, columns=attrs, normalize_=True)
    # load example image
    img, attrs_ = train_set[idx]
    i = attrs_[attrs.index('intensity')]
    t = attrs_[attrs.index('thickness')]
    us = scm.encode(image = img.view(1, 1, 28, 28).repeat(rg.shape[0], 1, 1, 1), 
                    intensity = i.view(-1, 1).repeat(rg.shape[0], 1), thickness = t.view(-1, 1).repeat(rg.shape[0], 1))
    # backtrack entire range of thicknesses at once
    us_ast = backtrack_gradient(scm, vars_=['intensity'], vals_ast=rg, dist_fun="l4", verbose=True, lr=0.01, num_it=10000, **us)
    # plot distribution of latent space in background (standard multivariate normal)
    latents = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=10000)
    # contour plot of latent space
    sns.kdeplot(x=latents[:, 0], y=latents[:, 1], bw=1, color=list(plt.rcParams['axes.prop_cycle'])[1]['color'], levels=8, fill=True, thresh=0.15, alpha=0.5)
    plt.scatter(us_ast["intensity"], us_ast["thickness"], c=list(plt.rcParams['axes.prop_cycle'])[3]['color'], s=25)
    plt.plot(us["intensity"], us["thickness"], 'o', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
    plt.xlim(-3, 3)  # Set x-axis range from -3 to 3
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal')
    #plt.colorbar(location='left')
    #tikzplotlib.save("./morphomnist/visualizations/visualize_uiast_to_utast.tex", extra_axis_parameters=['axis equal image'])
    plt.show()

if __name__ == "__main__":
    main("./morphomnist/data", idx=5)
