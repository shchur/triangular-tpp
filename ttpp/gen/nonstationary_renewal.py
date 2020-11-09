import numpy as np
from pathlib import Path
from scipy.stats import gamma
from scipy.optimize import brentq
dataset_dir = Path(__file__).parents[2] / 'data'

def generate(max_time, n_sequences, filename='nonstationary_renewal'):
    times, nll = [], []

    for _ in range(n_sequences):
        L = max_time
        amp = 0.99
        l_t = lambda t: np.sin(2*np.pi*t/L)*amp + 1
        l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos(2*np.pi*t1/L) )*amp + (t2-t1)

        T = []
        lpdf = []
        x = 0

        k = 4
        rs = gamma.rvs(k, size=max_time)
        lpdfs = gamma.logpdf(rs, k)
        rs = rs / k
        lpdfs = lpdfs + np.log(k)

        for i in range(max_time):
            x_next = brentq(lambda t: l_int(x,t) - rs[i], x, x + max_time)
            l = l_t(x_next)
            T.append(x_next)
            lpdf.append(lpdfs[i] + np.log(l))
            x = x_next

        T = np.array(T)

        T = T[T < max_time]
        lpdf = np.array(lpdf)[:len(T)]
        score = -lpdf.sum()

        times.append(T)
        nll.append(score)

    if filename is not None:
        mean_number_items = sum(len(t) for t in times) / len(times)
        nll = [n/mean_number_items for n in nll]
        np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)
    else:
        return times
