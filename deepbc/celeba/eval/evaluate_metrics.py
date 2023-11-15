from celeba.scm.model import CelebaSCM
from optim import backtrack_linearize
from celeba.baselines import tab_CE
from celeba.data.datasets import CelebaContinuous
from celeba.baselines import TwoCompSCM
from celeba.data.meta_data import attrs
from celeba.eval.metrics import identity, causal
import matplotlib.pyplot as plt
import torch

METHODS = ["DeepBC", "interventional", "tabular CE", "non-causal CE"]

def main(cont_attr_path="./celeba/data/predictions/preds.pt", sample_size=100):
    ids, causes = [], []
    # load true SCM
    scm = CelebaSCM()
    # load tabular SCMs
    #tabular_scms = {attr : TwoCompSCM(attr=attr) for attr in attrs}
    # load dummy SCM for non-causal baseline
    #nc_scm = {attr : TwoCompSCM(attr=attr) for attr in attrs}
    # load data set
    data = CelebaContinuous(cont_attr_path=cont_attr_path, as_dict=True)
    # take sample_size data points from data set
    xs = [data[i] for i in range(sample_size)]
    # losses
    ids, causes = {method : [] for method in METHODS}, {method : [] for method in METHODS}
    # iterate through data points and generate counterfactuals for each method
    for i in range(sample_size):
        # sample attribute
        attr = attrs[int(torch.randint(0, len(attrs), (1,)).item())]
        # sample antecedent value
        val_ast = torch.tensor([[torch.randn(1).item()]], dtype=torch.float32)
        # DeepBC 
        # transform to latent space
        us = scm.encode(**xs[i]) 
        #us_ast_back = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, **us)
        #xs_ast_back = scm.decode(**us_ast_back)
        #ids["DeepBC"].append(identity(xs[i], xs_ast))
        #causes["DeepBC"].append(causal(xs_ast, scm))

        # DeepBC with tabular CE baseline
        #xs_ast_obs = tab_CE(scm, vars_=[attr], vals_ast=val_ast, **us)

        # interventional counterfactual
        xs_int_ast = scm.decode(**us, repl={attr : val_ast})
        ids["interventional"].append(identity(xs[i], xs_int_ast))
        causes["interventional"].append(causal(xs_int_ast, scm))

        # DeepBC with non-causal baseline
        #nc_scm = TwoCompSCM(attr=attr)
        #us_cp = nc_scm.encode(**xs)
        #us_nc = {"image" : us_cp["image"], attr : torch.zeros_like(us_cp[attr])}
        #us_ast_nc = backtrack_gradient(nc_scm, vars_=[attr], vals_ast=val_ast, lambda_=1e3, lr=1e-1, num_it=600, **us_nc)
        #xs_ast_nc = nc_scm.decode(**us_ast_nc)
    print("Mean identity: ", torch.mean(torch.tensor(ids["interventional"])))
    print("Mean causal: ", torch.mean(torch.tensor(causes["interventional"])))

if __name__ == "__main__":
    main()