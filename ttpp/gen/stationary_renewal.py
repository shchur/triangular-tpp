import numpy as np
from pathlib import Path
from scipy.stats import lognorm
dataset_dir = Path(__file__).parents[2] / 'data'

def generate(max_time, n_sequences, filename='stationary_renewal'):
    times, nll = [], []

    for _ in range(n_sequences):
        s = np.sqrt(np.log(6*6+1))
        mu = -s*s/2
        tau = lognorm.rvs(s=s, scale=np.exp(mu), size=1000)

        lpdf = lognorm.logpdf(tau, s=s, scale=np.exp(mu))
        T = tau.cumsum()

        T = T[T < max_time]
        lpdf = lpdf[:len(T)]

        score = -np.sum(lpdf)

        times.append(T)
        nll.append(score)

    if filename is not None:
        mean_number_items = sum(len(t) for t in times) / len(times)
        nll = [n/mean_number_items for n in nll]
        np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)
    else:
        return times
