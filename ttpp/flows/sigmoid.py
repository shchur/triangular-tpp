import torch
import torch.nn
import torch.nn.functional as F

from .base import Flow
from ttpp.utils import clamp_preserve_gradients


__all__ = [
    'Sigmoid',
    'Logit'
]


class Sigmoid(Flow):
    def __init__(self):
        """Computes 1/(1+e^(-x))
        """
        super().__init__()
        self.eps = 1e-6

    def forward(self, x):
        y = torch.sigmoid(x)
        logdet = - F.softplus(-x) - F.softplus(x)
        return y, logdet

    @torch.jit.export
    def inverse(self, y):
        y = clamp_preserve_gradients(y, self.eps, 1-self.eps)
        x = torch.log(y) - torch.log1p(-y)
        logdet = - torch.log(y) - torch.log1p(-y)
        return x, logdet


def Logit():
    """Computes ln(x/(1-x))
    """
    return Sigmoid().get_inverse()
