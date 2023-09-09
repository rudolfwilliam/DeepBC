"""This class integrates all individual generative models into a single SCM model."""

from celeba.scm.modules import CelebaCondVAE
from celeba.scm.modules import AttributeFlow
from scm.model import SCM
from json import load
from celeba.data.meta_data import attrs, graph_structure

class CelebaSCM(SCM):
    def __init__(self, ckpt_path="./celeba/scm/trained_models/checkpoints/",
                 config_path_flow="./celeba/scm/config/flow.json", 
                 config_path_vae="./celeba/scm/config/vae.json"):
        # read parameters from config file
        config_flow = load(open(config_path_flow, "r")) 
        models = {attr : AttributeFlow(name=attr, parents=graph_structure[attr], n_layers=config_flow["n_layers"], 
                                       linear_=bool(config_flow[attr + "_linear"])) for attr in attrs}
        config_vae = load(open(config_path_vae, "r"))
        models["image"] = CelebaCondVAE(n_chan=config_vae["n_chan"], latent_dim=config_vae["latent_dim"], 
                                        beta=config_vae["beta"], lr=config_vae["lr"], cond_dim=len(attrs))
        super(CelebaSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
