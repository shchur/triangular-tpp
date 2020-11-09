import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Flow


def _compute_knots(lengths, lower, upper):
    knots = torch.cumsum(lengths, dim=-1)
    knots = F.pad(knots, pad=(1, 0), mode='constant', value=0.0)
    knots = (upper - lower) * knots + lower
    knots[..., 0], knots[..., -1] = lower, upper  # This is expensive
    lengths = knots[..., 1:] - knots[..., :-1]
    return lengths, knots


class Spline(Flow):
    def __init__(self, 
        n_knots, 
        left=0., 
        right=1., 
        bottom=0., 
        top=1., 
        tails='undefined', 
        spline_order=2,
        min_bin_width = 1e-2,
        min_bin_height = 1e-2,
        min_derivative = 1e-2,
        min_lambda = 0.025,
        **kwargs
        ):
        """Rational linear/quadratic spline flow.

        Rational quadratic spline is based on https://github.com/bayesiains/nsf
        Rational linear spline is based on https://github.com/hmdolatabadi/LRS_NF

        We found RLS to provide no noticeable improvement, so we use RQS in our experiments.

        Args:
            n_knots: Number of knots for the spline.
            left: min input
            right: max input
            bottom: min output
            top: max output
            tails: behavior at tails either linear or undefined
            spline_order: either 1 or 2 - use rational linear or rational quadratic splines
            min_bin_width: minimum width of a spline segment
            min_bin_height: minimum height of a spline segment
            min_derivative: minimum derivative at a knot
            min_lambda: minimum value for lambda - only applies to rational linear splines
        """
        super().__init__()
        assert spline_order in [1, 2], 'Order rational polynomials of spline_order %i are not supported!' % spline_order
        assert tails in ['linear', 'undefined'], '%s tails are not supported!' % tails
        self.spline_order = spline_order
        self.tails = tails

        self.n_derivatives = n_knots + (1 if tails == 'undefined' else -1)
        self.n_knots = n_knots
        self.n_params = 2*self.n_knots + self.n_derivatives + (self.spline_order == 1) * self.n_knots
        self._params = nn.Parameter(torch.zeros(self.n_params))

        # We store everything as parameters to have it better accessible on the gpu
        self.left = nn.Parameter(torch.tensor(left), requires_grad=False)
        self.right = nn.Parameter(torch.tensor(right), requires_grad=False)
        self.bottom = nn.Parameter(torch.tensor(bottom), requires_grad=False)
        self.top = nn.Parameter(torch.tensor(top), requires_grad=False)

        self.min_bin_width = nn.Parameter(torch.tensor(min_bin_width), requires_grad=False)
        self.min_bin_height = nn.Parameter(torch.tensor(min_bin_height), requires_grad=False)
        self.min_derivative = nn.Parameter(torch.tensor(min_derivative), requires_grad=False)
        self.min_lambda = nn.Parameter(torch.tensor(min_lambda), requires_grad=False)

        self.min_d_constant = nn.Parameter(torch.log(torch.exp(1 - self.min_derivative) - 1), requires_grad=False)

        self.rest_width = nn.Parameter(1.0 - self.min_bin_width * self.n_knots, requires_grad=False)
        self.rest_height = nn.Parameter(1.0 - self.min_bin_height * self.n_knots, requires_grad=False)

        self.eps = torch.finfo(torch.get_default_dtype()).eps

        if self.min_bin_width * self.n_knots > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.min_bin_height * self.n_knots > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

    def reset_parameters(self, a=1.0):
        self._params.data.uniform_(-a, a)
        
    def forward(self, x):
        """Forward transformation.

        Args:
            x: Inputs, shape [batch_size, seq_len, 1]

        Returns:
            y: Outputs, shape [batch_size, seq_len, 1]
            log_det_jac: Log determinant of Jacobian, shape [batch_size, seq_len, 1]
        """
        y, log_det_jac = self._unconstrained_spline(x, self._params, inverse=False)
        return y, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        """Inverse transformation.

        Args:
            y: Inputs, shape [batch_size, seq_len, 1]

        Returns:
            x: Outputs, shape [batch_size, seq_len, 1]
            log_det_jac: Log determinant of Jacobian, shape [batch_size, seq_len, 1]
        """
        x, inv_log_det_jac = self._unconstrained_spline(y, self._params, inverse=True)
        return x, inv_log_det_jac

    def _unconstrained_spline(self, inputs, params, inverse: bool = False):
        if not inverse:
            inside_interval_mask = (inputs >= self.bottom) & (inputs <= self.top)
        else:
            inside_interval_mask = (inputs >= self.left) & (inputs <= self.right)
        outside_interval_mask = ~inside_interval_mask

        outputs = inputs.clone()
        logabsdet = torch.zeros_like(inputs)
        
        if self.tails == 'undefined' or inside_interval_mask.any():
            outputs[inside_interval_mask], logabsdet[inside_interval_mask] = self._rational_spline(
                inputs[inside_interval_mask],
                params[inside_interval_mask] if len(params.shape) > 1 else params,
                inverse=inverse
            )
        return outputs, logabsdet

    def _rational_spline(self, inputs, params, inverse: bool = False):
        # Decompose input
        width_logits = params[..., :self.n_knots]
        height_logits = params[..., self.n_knots:2*self.n_knots]
        derivative_logits = params[..., self.n_knots*2:self.n_knots*2+self.n_derivatives]

        # Append the fixed derivatives
        if self.tails == 'linear':
            derivative_logits = F.pad(derivative_logits, pad=(1, 1), mode='constant', value=self.min_d_constant)
        
        # Normalize widths and heights
        widths = self.min_bin_width + self.rest_width * F.softmax(width_logits, dim=-1)
        heights = self.min_bin_height + self.rest_height * F.softmax(height_logits, dim=-1)

        # Compute knots
        widths, cum_widths = _compute_knots(widths, self.left, self.right)
        heights, cum_heights = _compute_knots(heights, self.bottom, self.top)

        # Ensure positive derivatives
        derivatives = F.softplus(derivative_logits) + self.min_derivative

        # Find corresponding segments
        bin_idx = self._search_sorted(cum_heights if inverse else cum_widths, inputs)[..., None]

        # Ensure correct shapes
        n = len(bin_idx)
        widths = widths.expand(n, widths.shape[-1])
        cum_widths = cum_widths.expand(n, cum_widths.shape[-1])
        heights = heights.expand(n, heights.shape[-1])
        cum_heights = cum_heights.expand(n, cum_heights.shape[-1])
        derivatives = derivatives.expand(n, derivatives.shape[-1])

        # Select input data
        input_widths = widths.gather(-1, bin_idx)[..., 0]
        input_cum_widths = cum_widths.gather(-1, bin_idx)[..., 0]
        input_heights = heights.gather(-1, bin_idx)[..., 0]
        input_cum_heights = cum_heights.gather(-1, bin_idx)[..., 0]
        input_delta = input_heights / input_widths
        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        if self.spline_order == 1:
            # Prepare lambdas
            lambda_logits = params[..., self.n_knots*2+self.n_derivatives:]
            lambdas = (1 - 2*self.min_lambda) * torch.sigmoid(lambda_logits) + self.min_lambda
            lambdas = lambdas.expand(n, lambdas.shape[-1])
            input_lambdas = lambdas.gather(-1, bin_idx)[..., 0]
            
            result = self._rational_linear_spline(
                inputs=inputs, 
                x_k1_x_k=input_widths, 
                x_k=input_cum_widths, 
                y_k1_y_k=input_heights, 
                y_k=input_cum_heights,
                s_k=input_delta,
                d_k=input_derivatives,
                d_k1=input_derivatives_plus_one,
                lambdas=input_lambdas,
                inverse=inverse)

        elif self.spline_order == 2:
            result = self._rational_quadratic_spline(
                inputs=inputs, 
                x_k1_x_k=input_widths, 
                x_k=input_cum_widths,
                y_k1_y_k=input_heights, 
                y_k=input_cum_heights,
                s_k=input_delta,
                d_k=input_derivatives,
                d_k1=input_derivatives_plus_one,
                inverse=inverse)

        else:
            raise ValueError('Order %i rational polynomials are not support' % self.spline_order)
        return result

    def _rational_quadratic_spline(self,
            inputs,
            x_k1_x_k,
            x_k,
            y_k1_y_k,
            y_k,
            s_k,
            d_k,
            d_k1,
            inverse: bool = False):
        # Notation from https://arxiv.org/abs/1906.04032
        if inverse:
            y = inputs

            y_y_k = y - y_k

            dk1_dk_2sk = d_k1 + d_k - 2 * s_k
            y_y_k_dk1_dk_2sk = y_y_k * dk1_dk_2sk

            a = y_k1_y_k * (s_k - d_k) + y_y_k_dk1_dk_2sk
            b = y_k1_y_k * d_k - y_y_k_dk1_dk_2sk
            c = -s_k * y_y_k

            root = b.pow(2) - 4*a*c
            # assert (root >= 0).all()  # this requires waiting for cuda synchronization -> expensive

            xi = 2*c / (-b - torch.sqrt(root))
            outputs = xi * x_k1_x_k + x_k

            xi_inv_xi = xi * (1 - xi)
            xi2 = xi.pow(2)
            inv_xi2 = (1 - xi).pow(2)

            denominator = s_k + dk1_dk_2sk * xi_inv_xi
            derivative_numerator = s_k.pow(2) * (d_k1 * xi2 + 2 * s_k * xi_inv_xi + d_k * inv_xi2)

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, -logabsdet
        else:
            xi = (inputs - x_k) / x_k1_x_k
            xi_inv_xi = xi * (1 - xi)
            xi2 = xi.pow(2)
            inv_xi2 = (1 - xi).pow(2)

            numerator = y_k1_y_k * (s_k * xi2 + d_k * xi_inv_xi)
            denominator = s_k + (d_k1 + d_k - 2*s_k) * xi_inv_xi
            
            outputs = y_k + numerator / denominator

            derivative_numerator = s_k.pow(2) * (d_k1 * xi2 + 2 * s_k * xi_inv_xi + d_k * inv_xi2)
            logabsdet = torch.log(derivative_numerator) - 2*torch.log(denominator)
            return outputs, logabsdet

    def _rational_linear_spline(self, 
            inputs,
            x_k1_x_k,
            x_k,
            y_k1_y_k,
            y_k,
            s_k,
            d_k,
            d_k1,
            lambdas,
            inverse: bool = False,
            w_k: float = 1.):
        # Notation from https://arxiv.org/pdf/2001.05168.pdf
        inv_lambdas = 1. - lambdas

        w_k1 = torch.sqrt(d_k / d_k1) * w_k
        w_m = (lambdas * w_k * d_k + inv_lambdas * w_k1 * d_k1) * s_k
        
        y_k1 = y_k + y_k1_y_k
        y_m = (inv_lambdas * w_k * y_k + lambdas * w_k1 * y_k1) / (inv_lambdas * w_k + lambdas * w_k1)

        if inverse:
            y = inputs
            left_of_lambda = (y <= y_m).float()
            right_of_lambda = 1. - left_of_lambda

            # Lets cache some variables to avoid redundant computations
            y_k_y = y_k - y
            y_k1_y = y_k1 - y
            y_y_m = y - y_m
            y_k1_y_m = y_k1 - y_m
            y_m_y_k = y_m - y_k

            numerator = lambdas * w_k * y_k_y * left_of_lambda \
                + (lambdas * w_k1 * y_k1_y + w_m * y_y_m) * right_of_lambda
            
            denominator = (w_k * y_k_y + w_m * y_y_m) * left_of_lambda \
                + (w_k1 * y_k1_y + w_m * y_y_m) * right_of_lambda

            phi = numerator / denominator
            outputs = x_k1_x_k*phi + x_k

            derivative_numerator = lambdas * w_k * w_m * y_m_y_k * left_of_lambda \
                + inv_lambdas * w_m * w_k1 * y_k1_y_m * right_of_lambda

            derivative_numerator *= x_k1_x_k

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))
        else:
            x = inputs
            phi = (x - x_k) / x_k1_x_k
            left_of_lambda = (phi <= lambdas).float()
            right_of_lambda = 1. - left_of_lambda

            # Lets cache some variables to avoid redundant computations
            l_phi = lambdas - phi
            phi_l = phi - lambdas
            inv_phi = 1 - phi

            numerator = (w_k * y_k * l_phi + w_m * y_m * phi) * left_of_lambda \
                + (w_m * y_m * inv_phi + w_k1 * y_k1 * phi_l) * right_of_lambda

            denominator = (w_k * l_phi + w_m * phi) * left_of_lambda \
                + (w_m * inv_phi + w_k1 * phi_l) * right_of_lambda

            outputs = numerator/denominator

            derivative_numerator = lambdas * w_k * w_m * (y_m - y_k) * left_of_lambda \
                + (inv_lambdas * w_m * w_k1 * (y_k1 - y_m)) * right_of_lambda

            derivative_numerator /= x_k1_x_k
            
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(torch.abs(denominator))
        return outputs, logabsdet

    def _search_sorted(self, bin_locations, inputs):
        bin_locations[..., -1] += self.eps
        return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1
