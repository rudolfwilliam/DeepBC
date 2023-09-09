from scm.model import SCM
from scm.modules import StructuralEquation
from optim import backtrack_linearize
from utils import override
from celeba.baselines.sparsity_on_observed.train_regressor import Regressor
from celeba.data.meta_data import attrs
import torch

attr = "beard"

dummy_graph_structure = {**{attr_ : [] for attr_ in attrs if attr_ != attr},
                         attr : [attr_ for attr_ in attrs if attr_ != attr]}

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
        self.regressor = Regressor(ckpt_path="./celeba/baselines/sparsity_on_observed/trained_models/checkpoints/")
    
    def encode(self, x, cond):
        return self.regressor(cond) 
    
    def decode(self, u, cond):
        return self.regressor(cond)

class DummySCM(SCM):
    """Imitate counterfactual explanation methods without causal model."""
    def __init__(self, attr=["beard"], regressor_path="./celeba/baselines/sparsity_on_observed/trained_models/checkpoints/beard-epoch=05.ckpt"):
        self.attr = attr
        models = {attr_ : IDSE(name=attr_) for attr_ in attrs if attr_ != attr}
        models = {attr : LinearSE(name=attr), **models}
        self.ckpt_path = regressor_path
        self.graph_structure = dummy_graph_structure
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

def sparse_CE(scm, vars_, vals_ast, **us):
    # xs and us are identical
    xs = scm.decode(**us)
    xs_copy = xs.copy()
    xs_copy.pop("image")
    scm_attr = DummySCM(attr=vars_[0])
    xs_ast = backtrack_linearize(scm=scm_attr, vals_ast=vals_ast, vars_=vars_, sparse=True, n_largest=1, **xs_copy)
    xs_ast[attr] = scm_attr.models[attr].regressor(torch.cat([xs_ast[pa] for pa in scm_attr.graph_structure[attr]], dim=1))
    img_ast = scm.models["image"].decode(us["image"], torch.cat([xs_ast[pa] for pa in scm.graph_structure["image"]], dim=1))
    return {"image" : img_ast, **xs_ast}