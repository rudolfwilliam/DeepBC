from project_files.celeba.scm.modules import CelebaCondVAE
from project_files.celeba.scm.modules import AttributeFlow
from project_files.celeba.data.meta_data import attrs
from project_files.celeba.baselines.full_graph.meta_data import full_graph_structure
from deepbc import SCM
from json import load

class FullGraphCelebaSCM(SCM):
    def __init__(self):
        # read parameters from config file
        config_flow = load(open("./project_files/celeba/scm/config/flow.json", "r")) 
        models = {attr : AttributeFlow(name=attr, parents=full_graph_structure[attr], n_layers=config_flow["n_layers"]) for attr in attrs}
        # take original vae
        config_vae = load(open("./project_files/celeba/scm/config/vae.json", "r"))
        models["image"] = CelebaCondVAE(n_chan=config_vae["n_chan"], latent_dim=config_vae["latent_dim"], 
                                        beta=config_vae["beta"], lr=config_vae["lr"], cond_dim=len(attrs))
        super(FullGraphCelebaSCM, self).__init__(ckpt_path="./project_files/celeba/baselines/full_graph/trained_models/checkpoints/", 
                                                  graph_structure=full_graph_structure, **models)
                                                  