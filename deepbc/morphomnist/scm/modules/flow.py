from deepbc.src.deepbc.scm.modules import GCondFlow
from deepbc.src.deepbc.custom_components import CondFlow, SigmoidFlow, LogOddsFlow, ConstAddScaleFlow
from deepbc.src.deepbc.utils import override
from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
from normflows.flows import affine
import normflows as nf

class ThicknessFlow(GCondFlow):
    def __init__(self, name="thickness", n_layers=3, lr=1e-6, verbose=False):
        self.name = name
        super(ThicknessFlow, self).__init__(name, lr, verbose)
        base = nf.distributions.base.DiagGaussian(1)
        layers = [] 
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        layers.append(affine.coupling.AffineConstFlow((1,)))
        self.flow = nf.NormalizingFlow(base, layers)

    @override
    def forward(self, x, x_pa):
        return self.flow(x)

    @override    
    def encode(self, x, x_pa):
        return self.flow.inverse(x)

    @override 
    def decode(self, u, x_pa):
        return self.flow(u)
    
    @override
    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        loss = self.flow.forward_kld(x)
        self.log("train_loss", loss)
        return loss
    
    @override
    def validation_step(self, val_batch, batch_idx): 
        x, _ = val_batch
        loss = self.flow.forward_kld(x)
        self.log("val_loss", loss) 
        return loss


class WGThicknessFlow(GCondFlow):
    """Thickness flow with wrong graph structure. Conditional on intensity."""
    def __init__(self, name="thickness_wg", n_layers=3, lr=1e-6, verbose=False):
        self.name = name
        super(WGThicknessFlow, self).__init__(name, lr, verbose)
        base = nf.distributions.base.DiagGaussian(1)
        layers = [] 
        # flow is conditional on intensity
        layers.append(MaskedAffineAutoregressive(features=1, hidden_features=1, context_features=1))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        layers.append(LogOddsFlow())
        layers.append(ConstAddScaleFlow(const=10., scale=1/100))
        layers.append(affine.coupling.AffineConstFlow((1,)))
        self.flow = CondFlow(base, layers)

    
class IntensFlow(GCondFlow):
    def __init__(self, name="intensity", n_layers=3, lr=1e-6, verbose=False):
        self.name = name
        super(IntensFlow, self).__init__(name, lr, verbose)
        base = nf.distributions.base.DiagGaussian(1)
        layers = []
        # flow is conditional on thickness
        layers.append(MaskedAffineAutoregressive(features=1, hidden_features=1, context_features=1))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        layers.append(SigmoidFlow())
        # prevent log likelihood from being -infty at intialization
        layers.append(ConstAddScaleFlow(const=2., scale=1/5))
        layers.append(affine.coupling.AffineConstFlow((1,)))
        self.flow = CondFlow(base, layers)


class WGIntensFlow(ThicknessFlow):
    """Intensity flow with wrong graph structure. Unconditional. Same architecture as ThicknessFlow."""
    def __init__(self, name="intensity_wg", n_layers=3, lr=1e-6, verbose=False):
        super(WGIntensFlow, self).__init__(name, n_layers, lr, verbose)
