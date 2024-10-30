"""Create continuous attributes from the CelebA dataset."""

from project_files.celeba.data.meta_data import attrs
import torch


def main():
    # load data
    preds = torch.load("./project_files/celeba/data/predictions/preds.pt") 
    for i, _ in enumerate(attrs):
        preds[:, [i]] = (preds[:, [i]] - preds[:, [i]].mean()) / preds[:, [i]].std()
    torch.save(preds, "./project_files/celeba/data/predictions/preds.pt")

if __name__ == "__main__":
    main()
