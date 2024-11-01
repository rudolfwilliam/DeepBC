"""This class integrates all individual generative models into a single SCM model."""

from project_files.celeba.scm.modules import CelebaCondVAE
from project_files.celeba.scm.modules import AttributeFlow
from celeba.data.meta_data import attrs, graph_structure
from project_files.src.deepbc.scm.model import SCM
from json import load

class CelebaSCM(SCM):
    def __init__(self, ckpt_path="./project_files/celeba/scm/trained_models/checkpoints/",
                 config_path_flow="./project_files/celeba/scm/config/flow.json", 
                 config_path_vae="./project_files/celeba/scm/config/vae.json"):
        # read parameters from config file
        config_flow = load(open(config_path_flow, "r")) 
        models = {attr : AttributeFlow(name=attr, parents=graph_structure[attr], n_layers=config_flow["n_layers"], 
                                       n_hidden=config_flow[attr + "_n_hidden"], n_blocks=config_flow[attr + "_n_blocks"]) for attr in attrs}
        config_vae = load(open(config_path_vae, "r"))
        models["image"] = CelebaCondVAE(n_chan=config_vae["n_chan"], latent_dim=config_vae["latent_dim"], 
                                        beta=config_vae["beta"], lr=config_vae["lr"], cond_dim=len(attrs))
        super(CelebaSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
