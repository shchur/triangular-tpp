import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Flow
from .spline import Spline


class SplineRNN(Spline):
    def __init__(self, hidden_size, input_size=1, **kwargs):
        """RNN produces the parameter for a spline flow.

        Args:
            n_knots: Number of knots for the spline.
            hidden_size: Size of the RNN hidden state.
            in_features: Data dimension.
            min_value: Value below which the transformation is identity.
            max_value: Value above which the transformation is identity.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.n_params)
        del self._params

    def get_weights(self, x, inverse=False):
        """Obtain the transformation parameters using the RNN.

        Args:
            x: Inputs, shape [batch_size, seq_len, 1]

        Returns:
            unnorm_widths: shape [batch_size, seq_len, 1, n_knots]
            unnorm_heights: shape [batch_size, seq_len, 1, n_knots]
            unnorm_derivatives: shape [batch_size, seq_len, 1, n_knots + 1]
        """
        x = F.pad(x, (0, 0, 1, 0))[..., :-1, :]
        
        h, _ = self.rnn(x)

        params = self.linear(h).unsqueeze(-2)
        return params

    def forward(self, x):
        """Apply the forward transformation. Iteratively use previously
        generated outputs to update RNN and produce parameters of spline
        to inverse the next value.

        Args:
            x: Inputs, shape [batch_size, seq_len, 1]
        """

        y = torch.zeros(x.shape[0], 1, 1, device=x.device) # First result of inversion used to set spline params
        h = torch.zeros(1, x.shape[0], self.hidden_size, device=x.device) # First state

        inverses = []
        log_det_jac = []
        for i in range(x.shape[1]):
            # Update hidden state using previous inverse y
            _, h = self.rnn.forward(y, h)

            # Get spline params
            params = self.linear(h.transpose(0, 1)).unsqueeze(-2)
            
            # Get next inverse using next value of x
            y, ld = self._unconstrained_spline(
                x[:,i,None],
                params
            )

            inverses.append(y)
            log_det_jac.append(ld)

        inverses = torch.cat(inverses, 1)
        log_det_jac = torch.cat(log_det_jac, 1)
        return inverses, log_det_jac

    @torch.jit.export
    def inverse(self, y):
        """Apply the inverse transformation.

        Args:
            y: Inputs, shape [batch_size, seq_len, 1]
        """
        params = self.get_weights(y)
        x, inv_log_det_jac = self._unconstrained_spline(
            y,
            params,
            inverse=True
        )
        return x, inv_log_det_jac
