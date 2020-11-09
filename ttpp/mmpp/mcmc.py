# MCMC sampling for MMPP with Pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .utils import hard_step, soft_step


class MJPTrajectory:
    def __init__(self, arrival_times, states, t_max):
        if len(arrival_times) + 1 != states.shape[0]:
            raise ValueError("Shapes of t and z don't match")
        if states.ndim != 2:
            raise ValueError("States must be in one-hot format")
        self.n_intervals = len(arrival_times) + 1
        self.t = arrival_times
        self.tau = torch.zeros(self.n_intervals)
        self.tau[-1] = t_max
        self.tau[:-1] = arrival_times
        self.tau[1:] -= arrival_times
        self.z = states

    def __repr__(self):
        return f"MJPTrajectory({self.n_intervals})"


def stack_paths(samples):
    """Convert a list of MJPTrajectories into two tensors

    Args:
        samples: List of MJPTrajectory of length B

    Returns:
        tau: Dwell times, shape [B, N_max]
        z: States, shape [B, N_max, K]
    """
    n_max = max(s.n_intervals for s in samples)
    taus = []
    zs = []
    for s in samples:
        taus.append(F.pad(s.tau, (0, n_max - s.n_intervals)).unsqueeze(0))
        zs.append(F.pad(s.z, (0, 0, 0, n_max - s.n_intervals)).unsqueeze(0))
    tau = torch.cat(taus, axis=0)
    z = torch.cat(zs, axis=0)
    return tau, z


def combine_duplicates(tau, z):
    """Merge dwell times of repeating subsequent states."""
    different = torch.ones(z.shape[0]).bool()
    different[:-1] = torch.any(z[1:] != z[:-1], dim=1)
    z_new = z[different]
    tau_new = torch.zeros(different.sum().item())
    prev = 0
    for idx, nxt in enumerate(different.nonzero().view(-1)):
        nxt = nxt.item() + 1
        tau_new[idx] = tau[slice(prev, nxt)].sum()
        prev = nxt
    return tau_new, z_new


def sample_poisson(t_max, rate):
    """Sample inter-event times from a Poisson process.

    Args:
        t_max: Duration of the observed interval.
        rate: Rate of the Poisson process.

    Returns:
        t: Arrival times.
    """
    tau = []
    t_last = 0.0
    while True:
        tau_next = np.random.exponential(scale=1./rate)
        t_next = t_last + tau_next
        if t_next > t_max:
            break
        tau.append(tau_next)
        t_last += tau_next
    return np.array(tau).cumsum()


def assign_to_intervals_tau(tau, x, temp=None):
    """Assign events to intervals.

    Args:
        tau: sample from q(t), shape [batch_size, num_intervals]
        x: event times, shape [num_events, 1]
        temp: temperature for soft step. Default: None - use hard step.

    Returns:
        assignment: event to interval, shape [batch_size, num_events, num_intervals]
    """
    batch_size, num_intervals = tau.shape
    num_samples = x.shape[0]
    t = tau.cumsum(-1)[..., :-1]
    t.unsqueeze_(-2)
    if temp is None:
        step = hard_step
    else:
        step = lambda x: soft_step(x, temp)
    assignment = torch.zeros(batch_size, num_samples, num_intervals)
    assignment[..., -1] = 1.0
    diff = step(t - x)
    assignment[..., :-1] += diff
    assignment[..., 1:] -= diff
    return assignment


def sample_ihp(path, rates, omega, t_max):
    """Sample from an inhomogeneous Poisson process with pw-constant intensity.

    Sampling is done using thinning.

    Args:
        path: MJP trajectory (defines the pw-constant rates)
        rates: Rate of each state, shape [K]
        omega: Upper bound on the rates.
        t_max: End of the observed interval.

    Returns:
        t: Arrival times.
    """
    t = torch.Tensor(sample_poisson(t_max, omega))

    assignment = assign_to_intervals_tau(path.tau.view([1, -1]),
                                     t.view([-1, 1]))[0]
    real_intensity = (assignment @ path.z @ rates).view(-1)

    accept = (torch.rand_like(t) * omega) < real_intensity
    return t[accept]


def t_to_tau(t, t_max):
    tau = torch.zeros(len(t) + 1)
    tau[-1] = t_max
    tau[:-1] = t
    tau[1:] -= t
    return tau


def tau_to_t(tau):
    t = tau.cumsum(-1)[:-1]
    return t


def log_vector_matrix(vec, mtx):
    return torch.logsumexp(vec[..., np.newaxis] + mtx, dim=-2)


def log_matrix_vector(mtx, vec):
    return torch.logsumexp(mtx + vec[..., np.newaxis, :], dim=-1)


def ffbs_logspace(log_obs_prob, log_transition, log_prior):
    """Sample from the posterior of a Hidden Markov Model.

    Implemented according to Rao & Teh, JMLR 2014.
    All the computations are performed in log-space for numerical stability.

    Args:
        log_obs_prob: Log-likelihood matrix, shape [batch_size, n_intervals, n_states]
        log_transition: Log of transition matrix, shape [n_states, n_states]
        log_prior: Log of prior on z_0, shape [n_states]

    Returns:
        z: Samples of states given jump times, shape [batch_size, n_intervals, n_states]
    """
    n_intervals = log_obs_prob.shape[-2]
    n_states = log_obs_prob.shape[-1]
    log_alpha = torch.zeros_like(log_obs_prob)
    log_beta = torch.zeros_like(log_obs_prob)
    z = torch.zeros_like(log_obs_prob)
    log_alpha[..., 0, :] = log_prior
    for i in range(1, n_intervals):
        log_alpha[..., i, :] = log_vector_matrix(log_alpha[..., i - 1, :] + log_obs_prob[..., i - 1, :],
                                                 log_transition.T)
    log_beta[..., -1, :] = log_obs_prob[..., -1, :] + log_alpha[..., -1, :]
    probs = log_beta[..., -1, :].softmax(-1)
    z[..., -1, :] = F.one_hot(torch.multinomial(probs, 1), n_states)
    transition = log_transition.exp()
    for i in reversed(range(0, n_intervals - 1)):
        log_beta[..., i, :] = log_alpha[..., i, :] + log_obs_prob[..., i, :] + (z[..., i + 1, :] @ transition.T).log()
        probs = log_beta[..., i, :].softmax(-1)
        z[..., i, :] = F.one_hot(torch.multinomial(probs, 1), n_states)
    return z


def sample_mmpp_posterior(path_init, t_max, x, lmbd, Q, pi, omega, n_burn_in=100, n_sample=1000, tqdm=None):
    """Draw samples from the posterior distribution of a MMPP

    Args:
        path_init: Initial MJPTrajectory
        t_max: Duration of the observed interval.
        x: Observed events (realizations of the visible PP), shape [M, 1]
        lmbd: Rates of each state, shape [K]
        Q: Generator matrix, shape [K, K]
        pi: Prior distribution over z_0, shape [K]
        omega: Uniformization constant
        n_burn_in: Number of samples to discard.
        n_sample: Number of samples to return.
    """
    if omega <= Q.abs().diag().max():
        raise ValueError("Uniformization constant must be larger than the largest value of Q.")
    K = Q.shape[0]
    B = torch.eye(K) + Q / omega
    B = F.normalize(B, p=1, dim=1, eps=0.0)

    path = path_init
    samples = []

    if tqdm is None:
        range_iter = range
    else:
        range_iter = tqdm.trange

    for epoch in range_iter(n_burn_in + n_sample):
        jump_rates = omega + torch.diag(Q)
        oversampling = 2.0 * jump_rates.max().item()
        t_new = sample_ihp(path, jump_rates, oversampling, t_max)
        t = torch.sort(torch.cat([t_new, path.t])).values
        tau = t_to_tau(t, t_max)

        # Compute the likelihood matrix
        assignment = assign_to_intervals_tau(tau.view(1, -1), x)[0]
        counts = assignment.sum(0).view(-1, 1)  # num of events in each interval

        log_obs_prob = (counts * lmbd.log() - tau.view(-1, 1) * lmbd).unsqueeze(0)
        log_transition = B.log()
        log_prior = pi.log()
        z = ffbs_logspace(log_obs_prob, log_transition, log_prior)[0]

        tau_new, z_new = combine_duplicates(tau, z)
        t_new = tau_to_t(tau_new)

        path = MJPTrajectory(t_new, z_new, t_max)
        samples.append(path)
    return samples[n_burn_in:]


def m_step(tau, z, x, psi, alpha, beta):
    """Compute optimal values of lmbd, Q, pi, omega given samples from the posterior."""
    # avg number of transitions i->j
    trans_counts = (z[:, :-1].transpose(-2, -1) @ z[:, 1:]).mean(0)
    # avg total time spent in each of K states
    dwell_times = (z * tau.unsqueeze(-1)).sum(-2).mean(0)
    # avg number of observations in each of K states
    event_counts = (assign_to_intervals_tau(tau, x) @ z).sum(-2).mean(0)

    lmbd_new = event_counts / dwell_times
    psi = psi - 1
    pi_new = (z[:, 0, :].sum(0) + psi) / (z.shape[0] + psi.sum())

    Q_new = (trans_counts + alpha - 1) / (dwell_times.reshape([-1, 1]) + beta)
    Q_new[np.diag_indices_from(Q_new)] = 0.0
    Q_new -= torch.diag(Q_new.sum(1))

    omega_new = 2.0 * Q_new.diag().abs().max().item()

    return lmbd_new, Q_new, pi_new, omega_new
