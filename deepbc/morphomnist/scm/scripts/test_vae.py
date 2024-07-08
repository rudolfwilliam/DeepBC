from deepbc.morphomnist.scm.modules.vae import CondVAE
from deepbc.morphomnist.data.datasets import MorphoMNISTLike, normalize
from deepbc.morphomnist.scm.modules.flow import FlowThickness, FlowIntens
import matplotlib.pyplot as plt
import torch


def test_vae(data_dir):
    vae_chpt = "./morphomnist/trained_models/checkpoints/vae_img-epoch=25.ckpt"
    vae_img = CondVAE(cond_dim=2, latent_dim=16)
    vae_img.load_state_dict(torch.load(vae_chpt, map_location=torch.device('cpu'))["state_dict"])
    # load first data point
    data = MorphoMNISTLike(data_dir, train=True, columns=['intensity', 'thickness'])
    norm = normalize(data, load=True)
    data.images = norm[0]
    data.metrics['intensity'] = norm[1]
    data.metrics['thickness'] = norm[2]  
    intensity = data.metrics['intensity'][3]
    thickness = data.metrics['thickness'][3]
    img = data.images[3]
    metrics = torch.stack([intensity, thickness], dim=0)

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    # plot image
    plt.imshow(img, cmap='gray')
    # show reconstruction
    enc, _ = vae_img.encode(img.unsqueeze(0).unsqueeze(0), metrics.unsqueeze(0))
    xhat = vae_img.decode(enc, metrics.unsqueeze(0))
    fig.add_subplot(1, 2, 2)
    plt.imshow(xhat.squeeze().detach().numpy(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    test_vae()

