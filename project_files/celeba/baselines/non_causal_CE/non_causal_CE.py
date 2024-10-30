from project_files.celeba.baselines.non_causal_CE.vae import CelebaVAE
from project_files.celeba.data.modules import Classifier
from deepbc.utils import override
from deepbc import SCM
from deepbc.scm.modules import StructuralEquation
from json import load
import os
import torch


class ClassifierSE(StructuralEquation):
    def __init__(self, name, config):
        super(ClassifierSE, self).__init__()
        self.name = name
        self.classifier = Classifier(attr=name, n_chan=config["n_chan"])

    def encode(self, x, cond):
        return torch.zeros_like(x)

    def decode(self, u, cond):
        return self.classifier(cond)


class TwoCompSCM(SCM):
    """Imitates counterfactual explanation methods without causal model."""
    def __init__(self, config_path_vae="./project_files/celeba/baselines/non_causal_CE/config/vae.json",
                 config_path_cls="./project_files/celeba/data/config/classifier.json", 
                 ckpt_path="./project_files/celeba/baselines/non_causal_CE/trained_models/checkpoints/",
                 attr="beard"):
        
        self.graph_structure = {"image" : [], attr : ["image"]}
        # read parameters from config file
        config_vae = load(open(config_path_vae, "r"))
        vae = CelebaVAE(n_chan=config_vae["n_chan"], cond_dim=0, latent_dim=config_vae["latent_dim"])
        config_cls = load(open(config_path_cls, "r"))
        clsf = ClassifierSE(name=attr, config=config_cls)
        self.models = {attr : clsf, "image" : vae}
        self.ckpt_path = ckpt_path
        self.__load_parameters()
        self.__freeze_models()
    
    @override
    def __load_parameters(self):
        # load pre-trained model for first file name starting with model name
        for name, model in self.models.items():   
            file_name = next((file for file in os.listdir(self.ckpt_path) if file.startswith(name)), None)
            if name == "image":
                model.load_state_dict(torch.load(self.ckpt_path + file_name, map_location=torch.device('cpu'))["state_dict"])
            else:
                model.classifier.load_state_dict(torch.load(self.ckpt_path + file_name, map_location=torch.device('cpu'))["state_dict"])
    
    @override
    def __freeze_models(self):
        for name, model in self.models.items():
            if name == "image":
                for param in model.parameters():
                    param.requires_grad = False
            else:
                for param in model.classifier.parameters():
                    param.requires_grad = False
        