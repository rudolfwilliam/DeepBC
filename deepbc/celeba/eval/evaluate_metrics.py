from celeba.scm.model import CelebaSCM
from optim import backtrack_gradient, backtrack_linearize
from celeba.baselines import tab_CE
from celeba.data.datasets import Statistics
from celeba.baselines import TwoCompSCM, WrongGraphCelebaSCM
from celeba.data.meta_data import attrs, vars
from celeba.eval.metrics import obs, plausible, causal
import matplotlib.pyplot as plt
import torch

METHODS = ["DeepBC", "interventional", "tabular CE", "non-causal CE", "wrong graph"]

def main(cont_attr_path="./celeba/data/predictions/preds.pt", sample_size=500):
    torch.manual_seed(0)
    # load true SCM
    scm = CelebaSCM()
    # load data set
    stats = Statistics()
    # load dummy SCM for non-causal baseline
    nc_scms = {attr : TwoCompSCM(attr=attr) for attr in attrs} 
    # take sample_size data points from data set
    scm_wg = WrongGraphCelebaSCM()
    # losses
    ids, plauses, causes = {method : [] for method in METHODS}, {method : [] for method in METHODS}, {method : [] for method in METHODS}
    # iterate through data points and generate counterfactuals for each method
    for i in range(sample_size):
        # sample data point
        xs, us = scm.sample(std=1)
        # sample attribute
        attr = attrs[int(torch.randint(0, len(attrs), (1,)).item())]
        # sample antecedent value
        val_ast = torch.tensor([[torch.randn(1).item()]], dtype=torch.float32)
        # DeepBC 
        # make num_it a bit smaller than default because it takes a bit long
        us_ast_back = backtrack_linearize(scm, vars_=[attr], vals_ast=val_ast, num_it=10, **us)
        xs_ast_back = scm.decode(**us_ast_back)
        ids["DeepBC"].append(obs(xs, xs_ast_back))
        plauses["DeepBC"].append(plausible(xs_ast_back, scm))
        causes["DeepBC"].append(causal(xs, xs_ast_back, scm))

        # tabular CE baseline
        xs_ast_obs = tab_CE(scm, vars_=[attr], vals_ast=val_ast, sparse=False, **us)
        ids["tabular CE"].append(obs(xs, xs_ast_obs))
        plauses["tabular CE"].append(plausible(xs_ast_obs, scm))
        causes["tabular CE"].append(causal(xs, xs_ast_obs, scm))

        # interventional counterfactual
        xs_int_ast = scm.decode(**us, repl={attr : val_ast})
        ids["interventional"].append(obs(xs, xs_int_ast))
        plauses["interventional"].append(plausible(xs_int_ast, scm))
        causes["interventional"].append(causal(xs, xs_int_ast, scm))

        # DeepBC with non-causal baseline
        nc_scm = nc_scms[attr]
        # latent space will be "organized" differently, so we need to encode again
        us_cp = nc_scm.encode(**xs)
        us_nc = {"image" : us_cp["image"], attr : torch.zeros_like(us_cp[attr])}
        ## other approaches use gradient-based optimization, so do we here for fair comparison
        us_ast_nc = backtrack_gradient(nc_scm, vars_=[attr], vals_ast=stats.destandardize(attr, val_ast), 
                                       lambda_=1e3, lr=1e-1, num_it=800, **us_nc)
        xs_ast_nc = nc_scm.decode(**us_ast_nc)
        # get predictions from classifiers (and standardize)
        attrs_preds = {**{attr_ : stats.standardize(attr_, nc_scms[attr_].models[attr_].classifier(xs_ast_nc["image"])) for attr_ in attrs}, 
                       "image" : xs_ast_nc["image"]}
        ids["non-causal CE"].append(obs(xs, attrs_preds))
        plauses["non-causal CE"].append(plausible(attrs_preds, scm))
        causes["non-causal CE"].append(causal(xs, attrs_preds, scm))

        # sparse DeepBC with wrong graph
        us_wg = scm_wg.encode(**xs)
        # key order needs to be ordered according to wrong graph structure
        us_wg = {key : us_wg[key] for key in scm_wg.graph_structure.keys()}
        us_ast_wg_ord = backtrack_linearize(scm_wg, vars_=[attr], vals_ast=val_ast, num_it=10, **us_wg)
        xs_ast_wg = scm_wg.decode(**us_ast_wg_ord)
        xs_ast_wg = {key : xs_ast_wg[key] for key in vars}
        ids["wrong graph"].append(obs(xs, xs_ast_wg))
        plauses["wrong graph"].append(plausible(xs_ast_wg, scm))
        causes["wrong graph"].append(causal(xs, xs_ast_wg, scm))
        if i % 10 == 0:
            torch.save(ids, "ids.pt")
            torch.save(plauses, "plauses.pt")
            print(i)
    # save results
    torch.save(ids, "ids.pt")
    torch.save(plauses, "plauses.pt")

if __name__ == "__main__":
    main()
