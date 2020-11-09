import torch
import torch.nn as nn


class Flow(nn.Module):
    """Base class for transforms with learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Compute f(x) and log_abs_det_jac(x)."""
        raise NotImplementedError

    @torch.jit.export
    def inverse(self, y):
        """Compute f^-1(y) and inv_log_abs_det_jac(y)."""
        raise NotImplementedError

    def get_inverse(self):
        """Get inverse transformation."""
        return InverseFlow(self)


class InverseFlow(Flow):
    def __init__(self, base_flow):
        super().__init__()
        self.base_flow = base_flow
        if hasattr(base_flow, 'domain'):
            self.codomain = base_flow.domain
        if hasattr(base_flow, 'codomain'):
            self.domain = base_flow.codomain

    def forward(self, x):
        return self.base_flow.inverse(x)

    @torch.jit.export
    def inverse(self, y):
        return self.base_flow.forward(y)
    
    def get_inverse(self):
        return self.base_flow
