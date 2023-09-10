"""Conditional flows for modeling the continuous attributes of CelebA."""

from scm.modules import GCondFlow
from custom_components import CondFlow
from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
import normflows as nf
from normflows.flows.affine.coupling import AffineConstFlow

class AttributeFlow(GCondFlow):
    def __init__(self, name, parents, n_layers=10, linear_=True, lr=1e-3):
        super(AttributeFlow, self).__init__(name, lr)
        self.parents = parents
        base = nf.distributions.base.DiagGaussian(1)
        layers = []
        layers.append(AffineConstFlow((1,)))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        # flow is conditional on parents
        if linear_:
            layers.append(MaskedAffineAutoregressive(features=1, hidden_features=1, num_blocks=0, context_features=len(parents)))
        else:
            layers.append(MaskedAffineAutoregressive(features=1, hidden_features=1, num_blocks=2, context_features=len(parents)))
        self.flow = CondFlow(base, layers)
        