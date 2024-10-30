from json import load
from deepbc import SCM
from project_files.morphomnist.scm.modules import MmnistCondVAE, WGIntensFlow, WGThicknessFlow
from project_files.morphomnist.baselines.wrong_graph.meta_data import wrong_graph_structure, attrs

class WGMmnistSCM(SCM):
    def __init__(self, ckpt_path="./project_files/morphomnist/baselines/wrong_graph/trained_models/checkpoints/",
                 config_path_flow="./project_files/morphomnist/baselines/wrong_graph/config/flow.json", 
                 config_path_vae="./project_files/morphomnist/scm/config/vae.json"):
        config_flow = load(open(config_path_flow, "r"))
        config_vae = load(open(config_path_vae, "r"))
        models = {"thickness" : WGThicknessFlow(name="thickness", n_layers=config_flow["n_layers_thickness"]), 
                  "intensity" : WGIntensFlow(name="intensity", n_layers=config_flow["n_layers_intensity"]), 
                  "image" : MmnistCondVAE(cond_dim=len(attrs), latent_dim=config_vae["latent_dim"], 
                                          n_chan=config_vae["n_chan"], beta=config_vae["beta"], name="image")}
        super(WGMmnistSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=wrong_graph_structure, **models)
