from scm.model import SCM

class DiscreteCelebASCM(SCM):
    def __init__(self, ckpt_path, graph_structure, **models):
        def __init__(self, ckpt_path="./morphomnist/scm/trained_models/checkpoints/",
                 config_path_flow="./morphomnist/scm/config/flow.json", 
                 config_path_vae="./morphomnist/scm/config/vae.json"):
        config_flow = load(open(config_path_flow, "r"))
        config_vae = load(open(config_path_vae, "r"))
        models = {"thickness" : ThicknessFlow(name="thickness", n_layers=config_flow["n_layers_thickness"]), 
                  "intensity" : IntensFlow(name="intensity", n_layers=config_flow["n_layers_intensity"]), 
                  "image" : MmnistCondVAE(cond_dim=len(attrs), latent_dim=config_vae["latent_dim"], 
                                          n_chan=config_vae["n_chan"], beta=config_vae["beta"], name="image")}
        super(DiscreteCelebASCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
    

    def sample(self, num_samp=1, **xs):
        """Sample from the SCM."""
        us = self.encode(**xs

def main():
    # collect statistics from the data


if __name__ == "__main__":
    main()
