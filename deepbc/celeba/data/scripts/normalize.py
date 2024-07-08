"""Create continuous attributes from the CelebA dataset."""

from deepbc.celeba.data.meta_data import attrs
import torch


def main():
    # load data
    preds = torch.load("./celeba/data/predictions/preds.pt") 
    for i, _ in enumerate(attrs):
        preds[:, [i]] = (preds[:, [i]] - preds[:, [i]].mean()) / preds[:, [i]].std()
    torch.save(preds, "./celeba/data/predictions/preds.pt")

if __name__ == "__main__":
    main()
