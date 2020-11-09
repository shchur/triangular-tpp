import torch
import torch.nn as nn

from .base import Flow

class Affine(Flow):
    """Element-wise affine transformation y = ax + b.
    
    Args:
        scale_init: Initial value for the scale parameter a.
        shift_init: Initial value for the shift parameter b.
        use_shift: If False, shift parameter b is set to 0 and never used.
        trainable: Make the transformation parameters a and b learnable.
    """

    def __init__(self, scale_init=1.0, shift_init=0.0, use_shift=True, trainable=True):
        super().__init__()
        log_scale_init = torch.tensor([scale_init], dtype=torch.get_default_dtype()).log()
        self.log_scale = nn.Parameter(log_scale_init, requires_grad=trainable)
        if use_shift:
            shift_init = torch.tensor([shift_init], dtype=torch.get_default_dtype())
            self.shift = nn.Parameter(shift_init, requires_grad=trainable)
        else:
            self.shift = 0.0

    def forward(self, x):
        y = torch.exp(self.log_scale) * x + self.shift
        log_det_jac = self.log_scale.expand(y.shape)
        return y, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        x = (y - self.shift) * torch.exp(-self.log_scale)
        inv_log_det_jac = -self.log_scale.expand(x.shape)
        return x, inv_log_det_jac
