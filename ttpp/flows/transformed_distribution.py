import math
import numpy as np
import torch
import torch.nn as nn
import warnings

from typing import List, Union

from .base import Flow
from .rnn import SplineRNN


__all__ = [
    'TransformedExponential'
]


class StandardExponential(nn.Module):
    """Standard exponential distribution with unit rate."""
    def __init__(self):
        super().__init__()

    def log_survival(self, x):
        result = -x
        result[x < 0] = 0.0
        return result

    def sample(self, sample_shape: List[int], device: str):
        return torch.empty(sample_shape, device=device).exponential_()

    def rsample(self, sample_shape: List[int], device: str):
        return self.sample(sample_shape, device)


class TransformedExponential(nn.Module):
    """Base class for all the models considered in the paper (except Hawkes process).

    We parametrize TPP densities by specifying the sequence of transformations that 
    convert an arbitrary TPP into a homogeneous Poisson process with unit rate.

    There are two small differences compared to the notation used in the paper:
    1. This class defines the inverse transformation (F^{-1} in the paper) that converts
      an HPP sample into our target TPP.
    2. To avoid redundant computations, our base distribution is product of iid unit
      exponential distriubitons. This corresponds to modeling the inter-event times of an HPP.
      We would obtain the HPP arrival times if apply cumulative sum. Since the log-determinant
      of cumulative sum is equalto zero, this doesn't change the results.
    
    Args:
        transforms: List of transformations applied to the distribution.
        t_max: Duration of the interval on which the TPP is defined (denoted as T in the paper).
    """
    def __init__(self, transforms: Union[List[Flow], Flow], t_max: float):
        super().__init__()
        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList([transforms, ])
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
            self.inv_transforms = nn.ModuleList(transforms[::-1])
        else:
            raise ValueError(f"transforms must a Flow or a list, but was {type(transforms)}")
        self.n_transforms = len(self.transforms)
        self.base_dist = StandardExponential()
        self.t_max = t_max

    @property
    def is_recurrent(self):
        """Indicates whether this model uses a RNN.

        Returns:
            bool: True if an RNN layer is used
        """
        return any(type(t) == SplineRNN for t in self.transforms)

    def forward(self, z):
        """Transforms z by F.

        Args:
            z: Padded sample, shape [batch_size, seq_len, 1]

        Returns:
            torch.Tensor: F(z), same shape as z
        """
        # TODO: Implement caching of log_det_jacobian
        for transform in self.transforms:
            z, log_det_jacobian = transform.forward(z)
        return z

    @torch.jit.export
    def inverse(self, x):
        """Transforms x by the inverse of F.

        Args:
            x: Padded sample, shape [batch_size, seq_len, 1]

        Returns:
            torch.Tensor: F^-1(x), same shape as x
        """
        for transform in self.inv_transforms:
            x, log_det_jacobian = transform.inverse(x)
        return x

    @torch.jit.export
    def log_prob(self, x, mask):
        """Compute log probability of a batch of sequences.

        Args:
            x: Padded sample, shape [batch_size, seq_len, 1]
            mask: Boolean mask indicating which entries of x correspond to 
                observed events (i.e. not padding), shape [batch_size, seq_len, 1]

        Returns:
            log_p: Log density for each sequence, shape [batch_size]
        """
        log_intensity = 0.0
        for i, transform in enumerate(self.inv_transforms):
            x, inv_log_det_jacobian = transform.inverse(x)
            log_intensity += inv_log_det_jacobian
        
        log_survival = self.base_dist.log_survival(x)
        # Mask values that shouldn't be computed
        if mask is not None:
            log_intensity = log_intensity * mask
        result = (log_intensity + log_survival).squeeze(-1).sum(-1)
        return result

    @torch.jit.export
    def rsample_n(self, batch_size: int, seq_len: int, device: str='cuda:0'):
        """Draw a sequences of given length from the transformed distribution.

        Returns:
            x: Sample, shape [batch_size, seq_len 1]
        """
        x = self.base_dist.rsample([batch_size, seq_len, 1], device)
        return self(x)

    @torch.jit.export
    def rsample(self, batch_size: int, device: str='cuda:0', init_seq_len: int = 100, seq_len_mult: float = 1.5):
        """Sample from the transformed distribution.

        This a wrapper around `rsample_n` that ensures that the event sequence
        is long enough to cover the [0, t_max] interval.

        Returns:
            x: Sample, shape [batch_size, seq_len 1]
        """
        long_enough = False
        seq_len = init_seq_len
        while not long_enough:
            z = self.base_dist.rsample([batch_size, seq_len, 1], device)
            x = self(z)
            # Make sure that the last event in each sequence happens after t_max
            long_enough = (x.max(-2).values.min() > self.t_max).item()
            if long_enough:
                return x
            else:
                warnings.warn(f"Generated sequence of length {seq_len} was too short. "
                               "You should probably increase init_seq_len")
                seq_len = int(seq_len_mult * seq_len)

    def init(self):
        """Initializes modules (required for RNN after loading)
        """
        for m in self.modules():
            if issubclass(type(m), torch.nn.RNNBase):
                m.flatten_parameters()
