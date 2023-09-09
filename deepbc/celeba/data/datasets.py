"""Custom dataset of CelebA with continuous attributes as generated by classifiers."""

from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop, Compose, ConvertImageDtype
import torch

DATA_PATH = "./celeba/data"

def load_data():
    transforms = Compose([CenterCrop(150), Resize((128, 128)), ToTensor(), ConvertImageDtype(dtype=torch.float32), Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),])
    data = CelebA(root=DATA_PATH, split='all', transform=transforms, download=True)
    return data

class CelebaContinuous(Dataset):
    def __init__(self, cont_attr_path, transform=None):
        super().__init__()
        self.transform = transform
        self.data = load_data()
        self.cont_attr = torch.load(cont_attr_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx][0], self.cont_attr[idx])
        return self.data[idx][0], self.cont_attr[idx]

# Custom transform to only select attributes
class SelectAttributesTransform:
    def __init__(self, attr_idx, pa_idx):
        self.attr_idx = attr_idx
        self.pa_idx = pa_idx

    def __call__(self, img, attrs):
        return attrs[[self.attr_idx]], torch.Tensor([attrs[idx] for idx in self.pa_idx])
