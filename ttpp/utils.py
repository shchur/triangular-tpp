import torch
import torch.nn.functional as F


def diff(x, dim: int = -1):
    """Inverse of x.cumsum(dim=dim).

    Compute differences between subsequent elements of the tensor.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def numpy(tensor):
    """Convert a torch.tensor to a 1D numpy.ndarray."""
    return tensor.cpu().detach().numpy().ravel()
