"""Plot images with different antecedent intensities for deep backtracking and interventional counterfactuals."""

from deepbc.optim import backtrack_linearize
import matplotlib.pyplot as plt
import torch
#import tikzplotlib
import numpy as np
from project_files.morphomnist.data.meta_data import attrs
from project_files.morphomnist.data.datasets import MorphoMNISTLike
from project_files.morphomnist.scm.model import MmnistSCM

rg = np.array([-2, -1.5, 0, 1.5, 2])

def main(data_dir, idx):
    scm = MmnistSCM()
    scm.eval()
    # load the training images
    data = MorphoMNISTLike(data_dir, train=True, columns=attrs)
    # load example image
    img, attrs_ = data[idx]
    i = attrs_[attrs.index('intensity')]
    t = attrs_[attrs.index('thickness')]
    us = scm.encode(image = img.view(1, 1, 28, 28), intensity = i.view(1, -1), thickness = t.view(1, -1))
    _, axs = plt.subplots(2, 5)
    for j, item in enumerate(rg):
        i_ast = torch.tensor(item, dtype=torch.float32).view(-1, 1)
        us_ast = backtrack_linearize(scm, vars_=["intensity"], vals_ast=i_ast, **us)
        xs_ast = scm.decode(**us_ast)
        print(xs_ast["thickness"])
        # create interventional counterfactual
        img_ast_int = scm.models["image"].decode(us["image"], torch.cat((i_ast.view(-1, 1), t.view(-1, 1)), dim=1))
        # plot images
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].imshow(xs_ast["image"].view(28, 28).detach().numpy(), cmap='gray', vmin=-0.4, vmax=4.45)
        axs[0, j].set_title(f"i_ast={item}")
        axs[1, j].set_yticklabels([])
        axs[1, j].set_xticklabels([])
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].imshow(img_ast_int.view(28, 28).detach().numpy(), cmap='gray', vmin=-0.4, vmax=4.45)
    plt.subplots_adjust(wspace=0, hspace=0)
    #tikzplotlib.save("./project_files/morphomnist/visualizations/tex_files/imgs_intensity.tex")
    #plt.savefig("./project_files/morphomnist/visualizations/img/imgs_intensity.pdf")
    plt.show()

if __name__ == "__main__":
    main("./project_files/morphomnist/data", idx=5)