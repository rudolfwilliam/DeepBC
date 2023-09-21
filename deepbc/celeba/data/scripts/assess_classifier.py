"""Visually assess the performance of the classifiers on a few images."""

from pytorch_lightning import Trainer
from celeba.scm.modules.classifier import Classifier
from celeba.data.datasets import load_data
from celeba.data.meta_data import attr2int
import torch
import matplotlib.pyplot as plt
import os


def main():
    torch.manual_seed(42)
    # Load the data
    data = load_data()
    idxs = range(390, 400)
    attr = "bald"
    # initialize models
    clsfier = Classifier(attr=attr2int[attr])
    file_name = next((file for file in os.listdir("./celeba/data/trained_models/classifiers/checkpoints/") if file.startswith(attr)), None)
    clsfier.load_state_dict(torch.load("./celeba/data/trained_models/classifiers/checkpoints/" + file_name, map_location=torch.device('cpu'))["state_dict"])
    fig, axs = plt.subplots(1, 10)
    for i, idx in enumerate(idxs):
        print(torch.sigmoid(clsfier(data[idx][0].unsqueeze(0))))
        axs[i].imshow(data[idx][0].permute(1, 2, 0))
        axs[i].set_title(round(clsfier(data[idx][0].unsqueeze(0)).item(), 2))
    plt.show()


if __name__ == "__main__":
    main()