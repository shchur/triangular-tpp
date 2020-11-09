import numpy as np
from pathlib import Path
dataset_dir = Path(__file__).parents[2] / 'data'


def simulate_hawkes(max_time, mu, alpha, beta):
    T = []
    LL = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential() / l
        x = x + step

        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0]*step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1]*step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0] * step)
        l_trg2 *= np.exp(-beta[1] * step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next / l:
            T.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if x > max_time:
                break

    T = np.array(T)
    T = T[T < max_time]

    score = -np.array(LL)[:len(T)].sum()

    return T, score


def _generate(max_time, n_sequences, filename, mu, alpha, beta):
    times, nll = [], []

    for _ in range(n_sequences):
        T, score = simulate_hawkes(max_time, mu, alpha, beta)
        times.append(T)
        nll.append(score)

    mean_number_items = sum(len(t) for t in times) / len(times)
    nll = [n/mean_number_items for n in nll]
    np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)


def generate(max_time, n_sequences, filename='hawkes'):
    if filename is not None:
        _generate(max_time, n_sequences, filename + '1', mu=0.2, alpha=[0.8,0.0], beta=[1.0,20.0])
        _generate(max_time, n_sequences, filename + '2', mu=0.2, alpha=[0.4,0.4], beta=[1.0,20.0])
    else:
        return times
