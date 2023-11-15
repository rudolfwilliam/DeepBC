from scm.model import SCM
from scm.modules import StructuralEquation
from optim import backtrack_linearize
from utils import override
from celeba.baselines.tabular.train_regressor import Regressor
from celeba.data.meta_data import attrs
import os
import torch

class IDSE(StructuralEquation):
    """Identity function."""
    def __init__(self, name):
        self.name = name
        super(IDSE, self).__init__()

    def encode(self, x, cond):
        return x
    
    def decode(self, u, cond):
        return u

class LinearSE(StructuralEquation):
    def __init__(self, name):
        self.name = name  
        super(LinearSE, self).__init__()
        self.regressor = Regressor(ckpt_path="./celeba/baselines/sparsity_on_observed/trained_models/checkpoints/", name=name)
    
    def encode(self, x, cond):
        return self.regressor(cond) 
    
    def decode(self, u, cond):
        return self.regressor(cond)

class DummySCM(SCM):
    """Imitate counterfactual explanation methods without causal model."""
    def __init__(self, graph_structure, attr="beard", regressor_path="./celeba/baselines/sparsity_on_observed/trained_models/checkpoints/beard-epoch=05.ckpt"):
        self.attr = attr
        models = {attr_ : IDSE(name=attr_) for attr_ in attrs if attr_ != attr}
        models = {attr : LinearSE(name=attr), **models}
        self.ckpt_path = regressor_path
        self.graph_structure = graph_structure
        self.models = models
        self.__load_parameters()
        # no need for training further
        self.__freeze_models()

    @override
    def __load_parameters(self):
        # load regressor only
        self.models[self.attr].regressor.load_state_dict(torch.load(self.ckpt_path, map_location=torch.device('cpu'))["state_dict"])
    
    @override 
    def __freeze_models(self):
        # freeze regressor only
        for param in self.models[self.attr].regressor.parameters():
            param.requires_grad = False

def tab_CE(scm, vars_, vals_ast, ckpt_path="./celeba/baselines/sparsity_on_observed/trained_models/checkpoints/", sparse=True, **us):
    # xs and us are identical
    dummy_graph_structure = {**{attr_ : [] for attr_ in attrs if attr_ != vars_[0]},
                                vars_[0] : [attr_ for attr_ in attrs if attr_ != vars_[0]]}
    xs = scm.decode(**us)
    xs_copy = xs.copy()
    xs_copy.pop("image")
    # find right path for regressor parameters
    file_name = next((file for file in os.listdir(ckpt_path) if file.startswith(vars_[0])), None)
    scm_attr = DummySCM(attr=vars_[0], graph_structure=dummy_graph_structure, regressor_path=ckpt_path + file_name)
    xs_ast = backtrack_linearize(scm=scm_attr, vals_ast=vals_ast, vars_=vars_, **xs_copy)
    xs_ast[vars_[0]] = scm_attr.models[vars_[0]].regressor(torch.cat([xs_ast[pa] for pa in scm_attr.graph_structure[vars_[0]]], dim=1))
    img_ast = scm.models["image"].decode(us["image"], torch.cat([xs_ast[pa] for pa in scm.graph_structure["image"]], dim=1))
    return {"image" : img_ast, **xs_ast}
