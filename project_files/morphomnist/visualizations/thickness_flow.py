"""Compare original thickness distribution to the thickness distribution obtained by sampling from the thickness flow."""

import torch
import matplotlib.pyplot as plt
from project_files.morphomnist.scm.model import SCM
from project_files.morphomnist.data.datasets import MorphoMNISTLike, normalize

def main(data_dir):
    scm = SCM()
    scm.eval()
    # Load the training images
    train_set = MorphoMNISTLike(data_dir, train=True, columns=['thickness', 'intensity'])
    train_set.images, train_set.metrics['intensity'], train_set.metrics['thickness'] = normalize(train_set)
    # obtain some samples from the thickness flow
    us = torch.randn(300, dtype=torch.float32).view(-1, 1)
    samples = scm.flow_thickness.decode(us)
    # plot the sample density
    plt.hist(samples.squeeze().detach(), bins=20)
    plt.hist(train_set[600:900]['thickness'], bins=20, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main("./project_files/morphomnist/data")
