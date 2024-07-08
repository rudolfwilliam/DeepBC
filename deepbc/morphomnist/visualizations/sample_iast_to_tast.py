"""Visualize the *antescedant i_ast* to t_ast, for multiple samples."""

from deepbc.morphomnist.data.datasets import MorphoMNISTLike
from deepbc.morphomnist.scm.model import MmnistSCM
from deepbc.morphomnist.data.meta_data import attrs
from deepbc.src.deepbc.sampling import langevin_mc
from deepbc.src.deepbc.optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

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
    us_asts = langevin_mc(scm, vars_=['intensity'], vals_ast=rg, step_size=1e-5, gap=1000, num_samp=400, **us)
    us_ast_back = backtrack_linearize(scm, vars_=['intensity'], vals_ast=rg, **us)
    xs_ast_back = scm.decode(**us_ast_back)
    # stack together
    us_ast_stacked = {key : torch.cat([sample[key] for sample in us_asts], dim=0) for key in scm.graph_structure.keys()}
    xs_ast_stacked = scm.decode(**us_ast_stacked)

    plt.plot(i, t, 'o', color=list(plt.rcParams['axes.prop_cycle'])[0]['color'])
    plt.gca().set_aspect('equal')  
    plt.show()
    torch.save(xs_ast_stacked, "./morphomnist/visualizations/xs_ast_stacked.pt")

if __name__ == "__main__":
    main("./morphomnist/data", idx=5)
    