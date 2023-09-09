from celeba.scm import CelebaSCM
from celeba.data.meta_data import attrs
import matplotlib.pyplot as plt
import torch

rg = torch.range(-3, 3, 0.5)

def main():
    # load scm
    scm = CelebaSCM()
    torch.manual_seed(1)
    # sample from scm
    xs, us = scm.sample(std=0)
    xs.pop("image")
    _, axs = plt.subplots(4, len(rg))
    # change values for attributes
    for j, attr in enumerate(attrs):
        for i, val in enumerate(rg):
            xs_temp = xs.copy()
            xs_temp[attr] = torch.Tensor([val]).view(-1, 1)
            xs_flat = torch.cat([x for x in xs_temp.values()], dim=1)
            img = scm.models["image"].decode(us['image'], xs_flat)
            axs[j, i].set_title(attr + f"={round(val.item(), 2)}")
            axs[j, i].imshow(img.squeeze().detach().permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
