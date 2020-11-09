import matplotlib.pyplot as plt
import numbers
import numpy as np
import torch
import seaborn as sns
sns.set_style('whitegrid', { 'axes.grid': False })


def plot_mmpp(events, states, durations, t_max):
    """Plot MMPP observations + states of the latent MJP."""

    plt.figure(figsize=[6, 2.5])

    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
        events = events.cpu().numpy()
        durations = durations.cumsum(0).detach().cpu().numpy()
        durations = np.concatenate([[0], durations], 0)

    for i in range(states.shape[0]):
        plt.fill_betweenx(np.array([0, 1]), durations[i], durations[i+1], color=f'C{states[i]}', alpha=0.2)
        plt.vlines(durations[i], 0, 1, color=f'C{states[i]}', linewidth=2, label=f'State {states[i]+1}')

    jitter = np.random.normal(loc=0.2, scale=0.05, size=len(events))
    plt.scatter(events, jitter, s=10, color='red', alpha=0.2, marker='o', label='True event')
    plt.title('True MJP trajectory')
    plt.xlim([0, t_max])
    plt.yticks([])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()
    plt.show()


def visualize_posterior(t, marginals, t_max, savefig=None, **figargs):
    """Visualize the posterior given samples."""

    plt.figure(figsize=[6, 2.5], **figargs)

    grid = torch.linspace(0, t_max - 0.01, 1000).view(-1, 1)
    assignment = assign_to_intervals(t, t_max, grid, temp=None)

    y = (assignment @ marginals).mean(0).squeeze().detach().cpu().numpy().cumsum(-1)
    y = np.concatenate([np.zeros([len(y), 1]), y], -1)
    x = grid.squeeze().detach().cpu().numpy()

    for i in range(y.shape[1] - 1):
        plt.fill_between(x, y[:,i], y[:,i+1], facecolor=f'C{i}', alpha=0.2)
    for i in reversed(range(y.shape[1] - 1)):
        plt.plot(x, y[:,i+1], color=f'C{i}', linewidth=2, label=f'State {i+1}')
    plt.xlim([0, t_max])
    plt.yticks([])
    plt.title('Posterior trajectory')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def log_vector_matrix(vec, mtx):
    """Vector-matrix multiplication in log-space.

    Args:
        vec: Vector, shape [..., ]
        mtx: Matrix, shape [..., ]

    Returns:
        result: Vector, shape [..., ]
    """
    return torch.logsumexp(vec[..., np.newaxis] + mtx, dim=-2)


def log_matrix_vector(mtx, vec):
    """Matrix-vector multiplication in log-space.
    Args:
        vec: Vector, shape [..., ]
        mtx: Matrix, shape [..., ]

    Returns:
        result: Vector, shape [..., ]
    """
    return torch.logsumexp(mtx + vec[..., np.newaxis, :], dim=-1)


def soft_step(x, temperature):
    """Differentiable relaxation of the Heaviside step function."""
    return torch.sigmoid(x / temperature)


def hard_step(x):
    """Heaviside step function."""
    return torch.sign(x).relu()


def clip_times(t_raw, t_max, temp=None):
    """Clip jump times at t_max.

    Example:
        t_max = 3.0
        t_raw = [1.0, 2.2, 3.5, 5.0]
        t, tau, inside_jump = clip_times(t_raw, t_max)
        # t = [1.0, 2.2, 3.0, 3.0]
        # tau = [1.0, 1.2, 0.8, 0.0, 0.0]
        # inside_jump = [1.0, 1.0, 0.0, 0.0]

    Args:
        t_raw: Raw jump times, shape [B, N, 1]
        t_max: Time of the end of the interval
        temp: Temperature for clipping. Default: None - use hard clipping.

    Returns
        t: Clipped jump times, shape [B, N, 1]
        tau: Dwell times, shape [B, N + 1, 1]
        inside_jump: Indicator mask, which jumps happened inside the [0, t_max] interval, shape [B, N, 1]
    """
    if t_raw.shape[-1] != 1:
        raise ValueError("t_raw shape must be [B, N, 1]")
    if temp is None:
        inside_jump = (t_raw < t_max).float()
    else:
        inside_jump = soft_step(t_max - t_raw, temp)
    t = t_raw.clamp(max=t_max)
    size_tau = list(t_raw.shape)
    size_tau[-2] += 1
    tau = torch.zeros(size_tau)
    tau[..., -1, :] = t_max
    tau[..., :-1, :] += t
    tau[..., 1:, :] -= t
    return t, tau, inside_jump


def assign_to_intervals(t, t_max, x, temp=None):
    """Assign events to intervals.

    Args:
        t: Jump times, shape [B, N, 1]
        t_max: Duration of the observed interval
        x: Event times, shape [M, 1]
        temp: Temperature for soft step. Default: None - use hard step.

    Returns:
        assignment: Event to interval assignment, shape [B, M, N + 1]
            number of intervals = N + 1
    """
    if isinstance(temp, numbers.Real) and (temp > 1 or temp <= 0):
        raise ValueError("Temperature must be in (0, 1] range")
    batch_size, num_jumps, _ = t.shape
    num_samples = x.shape[0]
    t = t.transpose(-2, -1)
    if temp is None:
        step = hard_step
    else:
        step = lambda x: soft_step(x, temp)
    assignment = torch.zeros(batch_size, num_samples, num_jumps + 1)
    assignment[..., -1] = step(t_max - x.view(-1))
    diff = step(t - x)
    assignment[..., :-1] += diff
    assignment[..., 1:] -= diff
    return assignment


def mask_diag(x):
    mask = torch.eye(x.shape[0]).bool()
    return x.masked_fill(mask, 0)
