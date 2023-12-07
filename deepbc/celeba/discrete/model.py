from discrete import IGR
from scm.model import SCM
from json import load
from celeba.data.meta_data import attrs, graph_structure

class CelebaDiscreteSCM(SCM):
    def __init__(self, ckpt_path="./celeba/discrete/trained_models/checkpoints/"):
        models = {attr : IGR(name=attr, parents=graph_structure[attr]) for attr in attrs}
        super(CelebaDiscreteSCM, self).__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)
        