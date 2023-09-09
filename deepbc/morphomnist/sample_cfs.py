from morphomnist.data.datasets import MorphoMNISTLike, normalize
from morphomnist.generate_bcf import generate_bcf
from torchvision.utils import save_image
import numpy as np
import torch

def sample_cfs(data_dir, var='intensity', num_samples=10):
    # load data set
    data = MorphoMNISTLike(data_dir, train=True, columns=['thickness', 'intensity'])
    data.images, data.metrics['intensity'], data.metrics['thickness'] = normalize(data)
    samples = data[:num_samples]
    perturbations = torch.distributions.Normal(torch.zeros((num_samples,)), 1).sample()
    xs = generate_bcf(samples["image"], samples["thickness"].view(-1, 1), samples["intensity"].view(-1, 1), perturbations.view(-1, 1), var)

    # save to file
    save_image(samples["image"].view(-1, 1, 28, 28), "./morphomnist/data/samples/images.png")
    np.savetxt("./morphomnist/data/samples/thicknesses.csv", samples["thickness"].numpy(), delimiter=',')
    np.savetxt("./morphomnist/data/samples/intensities.csv", samples["intensity"].numpy(), delimiter=',')
    save_image(xs["image"], "./morphomnist/data/samples/images_ast.png")
    np.savetxt("./morphomnist/data/samples/thickness_ast.csv", xs["thickness"].squeeze().numpy(), delimiter=',')
    np.savetxt("./morphomnist/data/samples/intensity_ast.csv", xs["intensity"].squeeze().numpy(), delimiter=',')