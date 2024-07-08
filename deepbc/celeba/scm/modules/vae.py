from deepbc.src.deepbc.scm.modules import CondVAE
from deepbc.src.deepbc.utils import flatten_list
from torch import nn
from collections import OrderedDict
import torch

class Encoder(nn.Module):
    def __init__(self, cond_dim, latent_dim=512, n_chan=[3, 128, 128, 128, 256, 512, 1024]):
        super().__init__()
        self.cond_dim = cond_dim
        # remove pointer from n_chan
        self.n_chan = n_chan.copy()
        self.n_chan[0] = self.n_chan[0] + cond_dim
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            OrderedDict(flatten_list([[
                ('enc' + str(i), nn.Conv2d(in_channels=self.n_chan[i], out_channels=self.n_chan[i+1], kernel_size=3, stride=2, padding=1)),
                ('enc' + str(i) + 'bn', nn.BatchNorm2d(self.n_chan[i+1])),
                ('enc' + str(i) + 'relu', nn.ReLU())] for i in range(len(self.n_chan)-1)])
            )
        )
        self.fc = nn.Linear(self.n_chan[-1]*4, self.latent_dim)
        # latent encoding
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)
    
    def __transform_cond(self, cond, size):
        # transform condition to match input size
        cond = cond.view(-1, self.cond_dim, 1, 1)
        cond = cond.repeat(1, 1, size, size)
        return cond

    def forward(self, x, cond):
        # concatenate condition with input image via broadcasting condition to image size
        cond = self.__transform_cond(cond, x.shape[2])
        x = torch.cat([x, cond], dim=1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        hidden = self.fc(x)
        # get distribution parameters
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)

        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, cond_dim, latent_dim=512, n_chan=[1024, 512, 256, 128, 128, 128, 3]):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim + self.cond_dim, self.n_chan[0]*4)
        # decoder
        self.conv1 = torch.nn.Sequential(
            OrderedDict(flatten_list([[
                ('dec' + str(i), nn.ConvTranspose2d(self.n_chan[i], self.n_chan[i+1], kernel_size=3, stride=2, padding=1, output_padding=1)),
                ('dec' + str(i) + 'bn', nn.BatchNorm2d(self.n_chan[i+1])),
                ('dec' + str(i) + 'relu', nn.ReLU())] for i in range(len(self.n_chan)-2)])
            )
        )
        # no batch norm and relu for last layer
        self.conv_fin = torch.nn.Sequential(
            OrderedDict([
                ('dec' + str(self.n_chan[-2]), nn.ConvTranspose2d(self.n_chan[-2], self.n_chan[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
            ])
        )

    def forward(self, u, cond):
        x = torch.cat([u, cond], dim=1)
        x = self.fc1(x)
        x = x.view(-1, self.n_chan[0], 2, 2)
        x = self.conv_fin(self.conv1(x))
        return x
    
class CelebaCondVAE(CondVAE):
    def __init__(self, cond_dim, n_chan=[3, 64, 64, 128, 256, 512, 1024], lr=1e-3, latent_dim=512, beta=4, name="image_vae"):
        # dimensionality of the conditional data
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.beta = beta

        encoder = Encoder(cond_dim, latent_dim, n_chan)
        decoder = Decoder(cond_dim, latent_dim, n_chan[::-1])

        super(CelebaCondVAE, self).__init__(encoder, decoder, latent_dim, beta=beta, lr=lr, name=name)
