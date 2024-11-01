"""This class integrates all individual generative models into a single SCM model."""

from json import load
from deepbc.scm.model import SCM
from project_files.morphomnist.scm.modules import MmnistCondVAE, ThicknessFlow, IntensFlow
from project_files.morphomnist.data.meta_data import graph_structure, attrs

class MmnistSCM(SCM):
    def __init__(self, ckpt_path="./project_files/morphomnist/scm/trained_models/checkpoints/",
                 config_path_flow="./project_files/morphomnist/scm/config/flow.json", 
                 config_path_vae="./project_files/morphomnist/scm/config/vae.json"):
        config_flow = load(open(config_path_flow, "r"))
        config_vae = load(open(config_path_vae, "r"))
        models = {"thickness" : ThicknessFlow(name="thickness", n_layers=config_flow["n_layers_thickness"]), 
                  "intensity" : IntensFlow(name="intensity", n_layers=config_flow["n_layers_intensity"]), 
                  "image" : MmnistCondVAE(cond_dim=len(attrs), latent_dim=config_vae["latent_dim"], 
                                          n_chan=config_vae["n_chan"], beta=config_vae["beta"], name="image")}
        super(MmnistSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
        