"""Conditional flows for modeling the continuous attributes of CelebA."""

from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
import normflows as nf
from normflows.flows.affine.coupling import AffineConstFlow
from deepbc.scm.modules import GCondFlow
from deepbc.custom_components import CondFlow

class AttributeFlow(GCondFlow):
    def __init__(self, name, parents, n_layers=10, n_hidden=1, n_blocks=0, lr=1e-3):
        super(AttributeFlow, self).__init__(name, lr)
        self.parents = parents
        base = nf.distributions.base.DiagGaussian(1)
        layers = []
        layers.append(AffineConstFlow((1,)))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        # flow is conditional on parents
        layers.append(MaskedAffineAutoregressive(features=1, num_blocks=n_blocks, hidden_features=n_hidden, context_features=len(parents)))
        self.flow = CondFlow(base, layers)
        