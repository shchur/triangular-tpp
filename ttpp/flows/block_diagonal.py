import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .base import Flow


class BlockDiagonal(Flow):
    def __init__(self, block_size, offset=0, inverse=True, **kwargs):
        """Applies repeated lower-triangular block diagonal matrices.
        Let H be the size of a lower-triangular block diagonal matrix A.
        This layer applies:
        [
            A, 0, 0, ...
            0, A, 0, ...
            0, 0, A, ...
            ., ., ., ...
        ]

        Args:
            block_size (int): Block size H
            offset (int, optional): Offset of A along the diagonal. Defaults to 0.
            inverse (bool, optional): Species which direction should be modelled directly. 
                The matrix for the inverse A^-1 is computed by torch.inv. Defaults to True.
        """
        super().__init__()
        self.block_size = block_size
        self.n_params = (block_size**2 - block_size) // 2 + block_size
        self.params = nn.Parameter(torch.zeros(self.n_params))
        mask = torch.zeros((block_size, block_size), dtype=torch.bool)
        idx = torch.tril_indices(block_size, block_size)
        mask[tuple(idx)] = 1
        self.mask = nn.Parameter(mask, requires_grad=False)
        diag_idx = (torch.arange(block_size) + 1).cumsum(0)  - 1
        self.diag_idx = nn.Parameter(diag_idx, requires_grad=False)
        self.offset = offset % self.block_size
        self._inverse = inverse


    def _logdet(self, x_shape: List[int], n_blocks: int, inverse: bool):
        """Computes the log|det(Jac(A(x)))|

        Args:
            x_shape (List[int]): shape of x
            n_blocks (int): Number of blocks A
            inverse (bool): Direction

        Returns:
            torch.Tensor: log|det(Jac(A(x)))|
        """
        logdet = self.params[None, None, self.diag_idx, None]\
            .expand(x_shape[0], n_blocks, self.block_size, x_shape[-1])\
            .reshape(x_shape[0], -1, x_shape[-1])[:, :x_shape[1]]
        if inverse is not self._inverse:
            return -logdet
        return logdet

    def _add_padding(self, x):
        """Applies padding to x such that x is a multiple of block_size

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Padded tensor
            int: Applied padding to the end
        """
        pad_end = self.block_size - ((x.shape[-2] + self.offset) % self.block_size)
        x = F.pad(x, (0, 0, self.offset, pad_end), 'constant', 0.)
        return x, pad_end

    def _compute_matrices(self, inverse: bool = False):
        """Returns the matrix A

        Args:
            inverse (bool, optional): Direction. Defaults to False.

        Returns:
            torch.Tensor: A
        """
        weights = self.params.clone()
        weights[self.diag_idx] = self.params[self.diag_idx].exp()

        mats = torch.zeros((self.block_size, self.block_size), device=weights.device)
        mats[self.mask] = weights
        if self._inverse is not inverse:
            mats = torch.inverse(mats)
        return mats

    def _matrices(self, inverse:bool = False):
        """Returns A or A^-1. If the matrix is not cached it is computed.

        Args:
            inverse (bool, optional): A or A^-1. Defaults to False.

        Returns:
            torch.Tensor: A^-inverse
        """
        # Use the cached matrix at evaluation
        # TODO: Caching
        return self._compute_matrices(inverse) 

    def _bmm(self, x, mat):
        """Batched matrix multiplication:
        X: n x a x b
        M: b x c
        returns n x a x c

        Args:
            x (torch.Tensor): Input tensor of shape batch_size,a,b
            mat (torch.Tensor): Matrix to apply of shape b,c

        Returns:
            torch.Tensor: Result of shape batch_size,a,c
        """
        return torch.einsum('ij,nkjl->nkil', mat, x)

    def forward(self, x):
        assert x.shape[1] >= self.block_size
        x, pad_end = self._add_padding(x)
        batch_size, inp_dim = x.shape[0], x.shape[-1]
        blocks = x.reshape(batch_size, -1, self.block_size, inp_dim)

        # Compute y
        mat = self._matrices(False)
        
        y = self._bmm(blocks, mat).reshape(batch_size, -1, inp_dim)
        # Get log det jac
        logdet = self._logdet(x.shape, blocks.shape[1], False)

        y, logdet = y[:, self.offset:-pad_end], logdet[:, self.offset:-pad_end]
        return y, logdet

    @torch.jit.export
    def inverse(self, y):
        assert y.shape[1] >= self.block_size
        y, pad_end = self._add_padding(y)
        batch_size, inp_dim = y.shape[0], y.shape[-1]
        blocks = y.reshape(batch_size, -1, self.block_size, inp_dim)

        # Compute x
        mat = self._matrices(True)
        x = self._bmm(blocks, mat).reshape(batch_size, -1, inp_dim)
        # Get log det jac
        logdet = self._logdet(y.shape, blocks.shape[1], True)

        x, logdet = x[:, self.offset:-pad_end], logdet[:, self.offset:-pad_end]
        return x, logdet
