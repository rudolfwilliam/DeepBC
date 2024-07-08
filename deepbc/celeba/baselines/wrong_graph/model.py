from deepbc.celeba.scm.modules import CelebaCondVAE
from deepbc.celeba.scm.modules import AttributeFlow
from deepbc.scm.model import SCM
from deepbc.celeba.data.meta_data import attrs
from deepbc.celeba.baselines.wrong_graph.meta_data import wrong_graph_structure
from json import load

class WrongGraphCelebaSCM(SCM):
    def __init__(self):
        # read parameters from config file
        config_flow = load(open("./celeba/scm/config/flow.json", "r")) 
        models = {attr : AttributeFlow(name=attr, parents=wrong_graph_structure[attr], n_layers=config_flow["n_layers"]) for attr in attrs}
        # take original vae
        config_vae = load(open("./celeba/scm/config/vae.json", "r"))
        models["image"] = CelebaCondVAE(n_chan=config_vae["n_chan"], latent_dim=config_vae["latent_dim"], 
                                        beta=config_vae["beta"], lr=config_vae["lr"], cond_dim=len(attrs))
        super(WrongGraphCelebaSCM, self).__init__(ckpt_path="./celeba/baselines/wrong_graph/trained_models/checkpoints/", 
                                                  graph_structure=wrong_graph_structure, **models)
        