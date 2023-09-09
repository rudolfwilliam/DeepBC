from scm.model import SCM
from celeba.baselines.non_causal_CE import CelebaVAE
from celeba.data.modules import Classifier
from scm.modules import StructuralEquation
from json import load
import torch


class ClassifierSE(StructuralEquation):
    def __init__(self, name, config, ckpt_path="./celeba/data/trained_models/data/trained_models/classifiers/checkpoints"):
        self.name = name
        self.classifier = Classifier(attr=name, n_chan=config["n_chan"])
        self.classifier.load_state_dict(torch.load(ckpt_path + name + ".pt", map_location=torch.device('cpu'))["state_dict"])
        super(ClassifierSE, self).__init__()

    def encode(self, x, cond):
        return torch.zeros_like(x)

    def decode(self, u, cond):
        return self.classifier(cond)


class TwoCompSCM(SCM):
    def __init__(self, config_path_vae="./celeba/scm/config/vae.json",
                 config_path_cls="./celeba/data/config/classifier.json", 
                 ckpt_path="./celeba/data/trained_models/classifiers/config/",
                 attr="beard"):
        """Imitates counterfactual explanation methods without causal model."""
        graph_structure = {"image" : [], attr : ["image"]}
        # read parameters from config file
        config_vae = load(open(config_path_vae, "r"))
        vae = CelebaVAE(n_chan=config_vae["n_chan"], latent_dim=config_vae["latent_dim"])
        config_cls = load(open(config_path_cls, "r"))
        clsf = ClassifierSE(name=attr, config=config_cls)
        models = {attr : clsf, "image" : vae}
        super(TwoCompSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
        