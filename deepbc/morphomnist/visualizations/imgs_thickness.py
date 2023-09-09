"""Plot images with different antecedent thicknesses for deep backtracking and interventional counterfactuals."""

import matplotlib.pyplot as plt
import torch
import tikzplotlib
import numpy as np
import seaborn as sns
from morphomnist.data.datasets import MorphoMNISTLike, normalize
from optim import backtrack_linearize
from morphomnist.scm.model import SCM

rg = np.arange(-1.5, 3.5, 1)

def main(data_dir, idx):
    scm = SCM()
    scm.eval()
    # load the training images
    data = MorphoMNISTLike(data_dir, train=True, columns=['intensity', 'thickness'])
    data.images, data.metrics['intensity'], data.metrics['thickness'] = normalize(data)
    # load example image
    example = data[idx]
    img = example['image']
    i = example['intensity']
    t = example['thickness']
    u_img, u_i, u_t = scm.encode(img.view(1, 1, 28, 28), i.view(1, -1), t.view(1, -1))
    _, axs = plt.subplots(2, 5)
    for j, item in enumerate(rg):
        t_ast = torch.tensor(item, dtype=torch.float32)
        u_img_ast, u_i_ast, u_t_ast = backtrack_linearize(u_img, u_i, u_t, t_ast, scm, var='thickness')
        img_ast, i_ast, t_ast = scm.decode(u_img_ast, u_i_ast.view(-1, 1), u_t_ast.view(-1, 1))
        # create interventional counterfactual
        img_ast_int = scm.models["image"].decode(u_img, torch.cat((scm.flow_intens.decode(torch.cat((u_i, t_ast.view(-1, 1)), dim=1)), t_ast.view(-1, 1)), dim=1))
        # plot images
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].imshow(img_ast.view(28, 28).detach().numpy(), cmap='gray', vmin=-0.4, vmax=3.6)
        axs[0, j].set_title(f"i_ast={item}")
        axs[1, j].set_yticklabels([])
        axs[1, j].set_xticklabels([])
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].imshow(img_ast_int.view(28, 28).detach().numpy(), cmap='gray', vmin=-0.4, vmax=3.6)
    plt.subplots_adjust(wspace=0, hspace=0)      
    plt.show()
    #tikzplotlib.save("./morphomnist/visualizations/tex_files/visualize_compare_cfs_intensity.tex")

if __name__ == "__main__":
    main("./morphomnist/data", idx=5)
