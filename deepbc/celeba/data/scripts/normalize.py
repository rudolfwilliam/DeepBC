"""Create continuous attributes from the CelebA dataset."""

import os
from celeba.data.meta_data import attrs
from celeba.scm.modules.classifier import Classifier
from celeba.data.datasets import load_data
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import torch


def main():
    # load data
    preds = torch.load("./celeba/data/predictions/preds.pt") 
    for i, _ in enumerate(attrs):
        preds[:, [i]] = (preds[:, [i]] - preds[:, [i]].mean()) / preds[:, [i]].std()
    torch.save(preds, "./celeba/data/predictions/preds.pt")

if __name__ == "__main__":
    main()
