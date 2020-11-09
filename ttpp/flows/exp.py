import numpy as np
import torch

from .base import Flow
from ttpp.utils import clamp_preserve_gradients


__all__ = [
    'Exp',
    'Log',
    'NegativeLog',
    'ExpNegative',
]


class Exp(Flow):
    """Convert samples as y = exp(x)."""

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        y = x.exp() #- self.epsilon
        log_det_jac = x
        return y, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        x = (y + self.epsilon).log()
        inv_log_det_jac = -y
        return x, inv_log_det_jac


def Log():
    """Convert samples as y = log(x)."""
    return Exp().get_inverse()


class NegativeLog(Flow):
    """Convert samples as y = -log(x)."""
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.get_default_dtype()).eps

    def forward(self, x):
        """
        y = -log(x)
        log |dy/dx| = -x
        """
        y = -torch.log(clamp_preserve_gradients(x, min=self.eps, max=1. - self.eps))
        log_det_jac = y
        return y, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        """
        x = exp(-y)
        log |dx/dy| = x
        """
        x = torch.exp(-y)
        inv_log_det_jac = -y
        return x, inv_log_det_jac


def ExpNegative():
    """Convert samples as y = exp(-x)."""
    return NegativeLog().get_inverse()
