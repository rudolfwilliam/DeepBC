"""Create continuous attributes from the CelebA dataset using classifier logits."""

from deepbc.celeba.data.meta_data import attrs
from deepbc.celeba.scm.modules.classifier import Classifier
from deepbc.celeba.data.datasets import load_data
from torch.utils.data import DataLoader
import os
import torch


def main(normalize=True, batch_size=64, ckpt_path="./celeba/data/trained_models/classifiers/checkpoints/"):
    # load data
    data = load_data()
    # store statistics for each attribute
    means = {}
    stds = {}
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    num_samples = len(data)
    preds = torch.Tensor(num_samples, len(attrs))
    # load classifiers
    for i, attr in enumerate(attrs):
        # load pre-trained model for first file name starting with flow.name
        file_name = next((file for file in os.listdir(ckpt_path) if file.startswith(attr)), None)
        clsfier = Classifier(attr)
        # we do this on cpu because we don't need to train the model
        clsfier.load_state_dict(torch.load(ckpt_path + file_name,
                                           map_location=torch.device("cpu"))["state_dict"])
        # save some memory for not computing gradients
        with torch.no_grad():
            # iterate over batches
            for (batch_idx, batch) in enumerate(data_loader):
                x, _ = batch
                batch_predictions = clsfier(x)  # Get predictions for the batch
                preds[batch_idx*batch_size:(batch_idx+1)*batch_size, [i]] = batch_predictions
            if normalize:
                means[attr] = preds[:, [i]].mean()
                stds[attr] = preds[:, [i]].std()
                preds[:, [i]] = (preds[:, [i]] - means[attr]) / stds[attr]
    # save predictions to file. These are the continuous attributes that will be used for training the SCM.
    #torch.save(preds, "./celeba/data/predictions/preds.pt")
    if normalize:
        # save statistics to file
        torch.save(means, "./celeba/data/predictions/means.pt")
        torch.save(stds, "./celeba/data/predictions/stds.pt")

if __name__ == "__main__":
    main()
