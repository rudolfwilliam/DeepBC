from celeba.scm import CelebaSCM
from celeba.data.meta_data import graph_structure
import matplotlib.pyplot as plt
import torch

rg = torch.arange(-3, 3, 0.5)

def main():
    # load scm
    scm = CelebaSCM()
    torch.manual_seed(24)
    # sample from scm
    xs, us = scm.sample()
    xs.pop("image")
    x = "age"
    y = "beard"
    xs_temp = xs.copy()
    for pa in graph_structure[y]:
        xs_temp[pa] = xs_temp[pa].repeat(rg.shape[0], 1)
    xs_temp[x] = rg.view(rg.shape[0], 1)
    xs_flat = torch.cat([xs_temp[k] for k in graph_structure[y]], dim=1)
    pred = scm.models[y].decode(us[y].repeat(rg.shape[0], 1), xs_flat)
    plt.scatter(xs_temp[x].squeeze(), pred.squeeze().detach())
    plt.show()


if __name__ == "__main__":
    main()