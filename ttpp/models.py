"""
All models are implemented like normalizing flow models: 
We specify TPP densities by defining the sequence of transformations that
map an arbitrary TPP sample into a vector, where each component follows
unit exponential distribution (which corresponds to the inter-event times
of a homogeneous Poisson process with unit rate). 

See `ttpp/flows/transformed_distribution.py` for more details.
"""
import torch
from . import flows


__all__ = [
    'InhomogeneousPoisson',
    'Renewal',
    'ModulatedRenewal',
    'Autoregressive',
    'TriTPP'
]


class InhomogeneousPoisson(flows.TransformedExponential):
    """Inhomogeneous Poisson process defined on the interval [0, t_max]"""
    def __init__(self, t_max, lambda_init=1.0, **kwargs):
        transforms = [
            flows.Cumsum(),
            flows.Affine(scale_init=(1. / lambda_init), use_shift=False),
            flows.Spline(**kwargs),
            flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
        ]
        super().__init__(transforms, t_max)


class Renewal(flows.TransformedExponential):
    """Renewal process, all inter-event times are sampled i.i.d."""
    def __init__(self, t_max, lambda_init=1.0, **kwargs):
        transforms = [
            flows.ExpNegative(),
            flows.Spline(**kwargs),
            flows.NegativeLog(),
            flows.Affine(scale_init=float(lambda_init), use_shift=False, trainable=True),
            flows.Cumsum(),
            flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False)
        ]
        super().__init__(transforms, t_max)


class ModulatedRenewal(flows.TransformedExponential):
    """Modulated renewal process - generalized Renewal and IhomogeneousPoisson."""
    def __init__(self, t_max, lambda_init=1.0, **kwargs):
        transforms = [
            flows.ExpNegative(),
            flows.Spline(**kwargs),
            flows.NegativeLog(),
            flows.Cumsum(),
            flows.Affine(scale_init=(1. / lambda_init), use_shift=False),
            flows.Spline(**kwargs),
            flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
        ]
        super().__init__(transforms, t_max)


class Autoregressive(flows.TransformedExponential):
    """RNN-based autoregressive model."""
    def __init__(self, t_max, lambda_init=1.0, **kwargs):
        transforms = [
            flows.ExpNegative(),
            flows.SplineRNN(**kwargs),
            flows.NegativeLog(),
            flows.Affine(scale_init=float(t_max / lambda_init), use_shift=False, trainable=True),
            flows.Cumsum(),
        ]
        super().__init__(transforms, t_max)


def triangular_layers(n_blocks, block_size, **kwargs):
    """Block-diagonal layers used in the TriTPP model."""
    result = []
    for i in range(n_blocks):
        offset = block_size //2 * (i%2)
        result.append(flows.BlockDiagonal(block_size=block_size, offset=offset, **kwargs))
    return result


class TriTPP(flows.TransformedExponential):
    """TriTPP model with learnable block-diagonal layers, generaled Modulated Renewal."""
    def __init__(self, t_max, lambda_init=1.0, **kwargs):
        transforms = [
            flows.ExpNegative(),
            flows.Spline(**kwargs),
            flows.Logit(),
        ] + triangular_layers(**kwargs) + [
            flows.Sigmoid(),
            flows.Spline(**kwargs),
            flows.NegativeLog(),
            flows.Cumsum(),
            flows.Affine(scale_init=(1. / lambda_init), use_shift=False, trainable=True),
            flows.Spline(**kwargs),
            flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
        ]
        super().__init__(transforms, t_max)


# def InhomogeneousPoisson(t_max, lambda_init=1.0, **kwargs):
#     """Inhomogeneous Poisson process defined on the interval [0, t_max]"""
#     transforms = [
#         flows.Cumsum(),
#         flows.Affine(scale_init=(1. / lambda_init), use_shift=False),
#         flows.Spline(**kwargs),
#         flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
#     ]
#     return flows.TransformedExponential(transforms)


# def Renewal(t_max, lambda_init=1.0, **kwargs):
#     """Renewal process, all inter-event times are sampled i.i.d."""
#     transforms = [
#         flows.ExpNegative(),
#         flows.Spline(**kwargs),
#         flows.NegativeLog(),
#         flows.Affine(scale_init=float(lambda_init), use_shift=False, trainable=True),
#         flows.Cumsum(),
#         flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False)
#     ]
#     return flows.TransformedExponential(transforms)


# def ModulatedRenewal(t_max, lambda_init=1.0, **kwargs):
#     """Modulated renewal process - generalized Renewal and IhomogeneousPoisson."""
#     transforms = [
#         flows.ExpNegative(),
#         flows.Spline(**kwargs),
#         flows.NegativeLog(),
#         flows.Cumsum(),
#         flows.Affine(scale_init=(1. / lambda_init), use_shift=False),
#         flows.Spline(**kwargs),
#         flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
#     ]
#     return flows.TransformedExponential(transforms)


# def Autoregressive(t_max=1.0, lambda_init=1.0, **kwargs):
#     """RNN-based autoregressive model."""
#     transforms = [
#         flows.ExpNegative(),
#         flows.SplineRNN(**kwargs),
#         flows.NegativeLog(),
#         flows.Affine(scale_init=float(t_max / lambda_init), use_shift=False, trainable=True),
#         flows.Cumsum(),
#     ]
#     return flows.TransformedExponential(transforms)


# def TriangularLayers(n_blocks, block_size, **kwargs):
#     """Block-diagonal layers used in the TriTPP model."""
#     result = []
#     for i in range(n_blocks):
#         offset = block_size //2 * (i%2)
#         result.append(flows.BlockDiagonal(block_size=block_size, offset=offset, **kwargs))
#     return result


# def TriTPP(t_max, lambda_init=1.0, **kwargs):
#     """TriTPP model with learnable block-diagonal layers."""
#     transforms = [
#         flows.ExpNegative(),
#         flows.Spline(**kwargs),
#         flows.Logit(),
#     ] + TriangularLayers(**kwargs) + [
#         flows.Sigmoid(),
#         flows.Spline(**kwargs),
#         flows.NegativeLog(),
#         flows.Cumsum(),
#         flows.Affine(scale_init=(1. / lambda_init), use_shift=False, trainable=True),
#         flows.Spline(**kwargs),
#         flows.Affine(scale_init=float(t_max), use_shift=False, trainable=False),
#     ]
#     return flows.TransformedExponential(transforms)
