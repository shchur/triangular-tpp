# Variational Expectation-Maximization
# Posterior over states, point estimate of parameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import assign_to_intervals, log_matrix_vector, log_vector_matrix, mask_diag


class ParamsMMPP(nn.Module):
    """Container for parameters of a Markov modulated Poisson process.

    Attributes:
        init: Distribution over the initial state (pi), shape [K]
        generator: Matrix of transition rates (A), shape [K, K]
        rates: Intensity (rate) of each MJP state (lambda), shape [K]
    """
    def __init__(self, n_states, avg_rate=1.0, trainable=False):
        super().__init__()
        self.n_states = n_states
        self.pre_generator = nn.Parameter(torch.empty(n_states, n_states), requires_grad=trainable)
        self.pre_rates = nn.Parameter(torch.empty(n_states), requires_grad=trainable)
        self.pre_log_init = nn.Parameter(torch.empty(n_states), requires_grad=trainable)
        self.reset_parameters()
        self.pre_rates.data += np.log(avg_rate)

    def reset_parameters(self):
        self.pre_generator.data.fill_(0.0)
        self.pre_rates.data = torch.linspace(0.1, 3., steps=self.n_states).log()
        self.pre_log_init.data.fill_(0.0)

    def set_parameters(self, init=None, generator=None, rates=None):
        if generator is not None:
            if generator.shape != torch.Size([self.n_states, self.n_states]):
                raise ValueError("Shape of generator must equal [n_states, n_states]")
            pre_gen = torch.Tensor(generator).abs().log()
            self.pre_generator.data = pre_gen
        if init is not None:
            if len(init) != self.n_states:
                raise ValueError("Shape of init must equal [n_states]")
            self.pre_log_init.data = torch.Tensor(init).log()
        if rates is not None:
            if len(rates) != self.n_states:
                raise ValueError("Shape of rates must equal [n_states]")
            self.pre_rates.data = torch.Tensor(rates).log()

    @property
    def log_init(self):
        return F.log_softmax(self.pre_log_init, dim=-1)

    @property
    def init(self):
        return self.log_init.exp()

    @property
    def log_generator(self):
        return self.pre_generator

    @property
    def generator(self):
        return self.pre_generator.exp()

    @property
    def log_transition(self):
        return self.log_generator - self.log_generator.logsumexp(-1, keepdims=True)

    @property
    def log_rates(self):
        return self.pre_rates

    @property
    def rates(self):
        return self.pre_rates.exp()



def map_loss(init, generator, rates, prior_init=None, prior_generator=None, prior_rates=None):
    """Compute regularization loss for the model parameters."""
    log_p = 0.0
    if prior_init is not None:
        log_p += prior_init.log_prob(init)
    if prior_generator is not None:
        log_p += prior_generator.log_prob(generator)
    if prior_rates is not None:
        log_p += prior_rates.log_prob(rates)
    return -log_p


def elbo_mjp(q_t, t, inside_jump, log_likelihood, log_marginals, state_transitions, params):
    """
    Args:
        q_t: Variational distribution over the jump times
        t: Jump times, shape [B, N, 1]
        inside_jump: Indicator - which jumps happened before T, shape [B, N, 1]
        log_likelihood: Log-likelihood of each state, shape [B, N + 1, K]
        log_marginals: Posterior marginal log-probas over states, shape [B, N + 1, K]
        state_transitions: Posterior transition probas between states, shape [B, N, K, K]
        params: Parameters of the MMPP, instance of ParamsMMPP
    """
    state_marginals = log_marginals.exp()
    elbo = (state_marginals[:, 0] @ params.log_init).mean(0)
    elbo += (state_marginals * log_likelihood).sum([-1, -2]).mean(0)

    trans_counts = (state_transitions * inside_jump.unsqueeze(-1)).sum(-3)
    elbo += (trans_counts * params.log_generator).sum([-1, -2]).mean(0)

    entropy = -q_t.log_prob(t, mask=inside_jump).mean(0)
    entropy -= (log_marginals[:, 1:, :] * state_marginals[:, 1:, :] * inside_jump).sum([-1, -2]).mean(0)
    entropy -= (log_marginals[:, 0, :] * state_marginals[:, 0, :]).sum(-1).mean(0)
    elbo += entropy
    return elbo


def forward_backward(log_likelihood, log_transition, log_init):
    """Perform inference over the states given the jump times.
    
    Args:
        log_likelihood: Log-likelihood of the observations for each state, shape [B, N + 1, K]
        log_transition: Log-transition-probas, shape [K, K]
        log_init: Log-probas of the initial state, shape [K]

    Returns:
        marginals: Posterior marginal probas, shape [B, N + 1, K]
        transition: Posterior transition probabilities, shape [B, N, K]
    """
    n_intervals = log_likelihood.shape[-2]
    n_states = log_likelihood.shape[-1]
    alpha = torch.zeros_like(log_likelihood)
    alpha[..., 0, :] = log_init + log_likelihood[..., 0, :]
    for i in range(1, n_intervals):
        alpha[..., i, :] = log_vector_matrix(alpha[..., i - 1, :], log_transition) + log_likelihood[..., i, :]
    beta = torch.zeros_like(alpha)
    for i in reversed(range(0, n_intervals - 1)):
        beta[..., i, :] = log_matrix_vector(log_transition, beta[..., i + 1, :] + log_likelihood[..., i + 1, :])
    # shape [batch_size, n_intervals, n_states]
    log_marginals = F.log_softmax(alpha + beta, dim=-1)
    log_transitions = (alpha.unsqueeze(-1)[:, :-1] + log_transition +
               log_likelihood.unsqueeze(-2)[:, 1:] + beta.unsqueeze(-2)[:, 1:])
    # shape [batch_size, n_intervals - 1, n_states, n_states]
    # perform softmax over dimensions [-2, -1]
    transitions = (log_transitions - log_transitions.logsumexp([-2, -1], keepdims=True)).exp()
    return log_marginals, transitions


def log_like_poisson(t, tau, t_max, events, rates, temp=None):
    """Log-likelihood of the observations given the latent state log p(o | s, t).

    log_obs_prob[i, j, k] = log p(observations in interval j | state[j] = k, jump times t[i]).
      
    Args:
        t: Jump times, shape [B, N, 1]
        tau: Dwell times, shape [B, N + 1, 1]
        x: Times of observed events, shape [M, 1]
        rates: Intensity (rate) of each MJP state, shape [K]
        temp: Temperature for relaxed assignment, None - use hard assignment

    Returns:
        log_obs_prob: Log-likelihood of observations, shape [B, N + 1, K]
    """
    assignment = assign_to_intervals(t, t_max, events, temp)
    counts = assignment.sum(-2)
    log_obs_prob = -tau * rates + counts.unsqueeze(-1) * rates.log()
    return log_obs_prob


def log_like_mjp(tau, generator):
    """Log-likelihood of the latent state.

    log_surv_prob[i, j, k] = log p(state[j] = k, jump times t[i]).
    
    Args:
        tau: Dwell times, shape [B, N + 1, 1]
        generator: Matrix of transition rates, shape [K, K]

    Returns:
        log_surv_prob: Log-likelihood of dwell times, shape [B, N + 1, K]
    """
    log_surv_prob = -tau * generator.sum(-1)
    return log_surv_prob
