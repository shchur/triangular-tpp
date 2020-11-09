import torch

from .base import Flow
from ttpp.utils import diff


class Cumsum(Flow):
    def __init__(self, dim=-2):
        """Compute cumulative sum along the specified dimension of the tensor.

        Args:
            dim: Dimension over which to perform the summation, -1 or -2.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = x.cumsum(self.dim)
        log_det_jac = torch.zeros_like(x)
        return y, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        x = diff(y, self.dim)
        inv_log_det_jac = torch.zeros_like(y)
        return x, inv_log_det_jac
