"""Assess conditional flow for intensity given thickness for varying thickness values."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from morphomnist.scm.model import SCM
from morphomnist.data.datasets import MorphoMNISTLike, normalize

def main(data_dir):
    scm = SCM()
    scm.eval()
    # Load the training images
    train_set = MorphoMNISTLike(data_dir, train=True, columns=['thickness', 'intensity'])
    train_set.images, train_set.metrics['intensity'], train_set.metrics['thickness'] = normalize(train_set)
    # Load example image
    example = train_set[5]
    img = example['image']
    thickness = example['thickness']
    intensity = example['intensity']
    u_x, u_i, u_t = scm.encode(img.view(1, 1, 28, 28), intensity.view(1, -1), thickness.view(1, -1))
    t_rg = np.arange(-2, 2, 0.2)
    i_s = []
    for t in t_rg:
        i = scm.flow_intens.decode(torch.cat([u_i.view(-1, 1), torch.tensor(t, dtype=torch.float32).view(-1, 1)], dim=1))
        i_s.append(i)
    plt.scatter(t_rg, torch.stack(i_s).squeeze().detach())
    plt.scatter(train_set[300:600]['thickness'], train_set[300:600]['intensity'], c='r')
    plt.show()

if __name__ == '__main__':
    main("./morphomnist/data")
    