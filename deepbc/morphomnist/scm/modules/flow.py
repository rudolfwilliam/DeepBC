from scm.modules import GCondFlow
from custom_components import CondFlow, SigmoidFlow, ConstAddScaleFlow
import normflows as nf
from utils import override
from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
from normflows.flows import affine

class ThicknessFlow(GCondFlow):
    def __init__(self, name="thickness", n_layers=3, lr=1e-6):
        self.name = name
        super(ThicknessFlow, self).__init__(name, lr)
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
        return loss
    
    @override
    def validation_step(self, val_batch, batch_idx): 
        x, _ = val_batch
        loss = self.flow.forward_kld(x)
        self.log("val_loss", loss) 
        return loss
    
class IntensFlow(GCondFlow):
    def __init__(self, name="intensity", n_layers=3, lr=1e-6):
        self.name = name
        super(IntensFlow, self).__init__(name, lr)
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