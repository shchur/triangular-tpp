import numpy as np
from pathlib import Path
dataset_dir = Path(__file__).parents[2] / 'data'

def generate(max_time, n_sequences, filename='self_correcting'):

    times, nll = [], []

    def self_correcting_process(mu,alpha,n):
        t = 0
        x = 0
        T = []
        log_l = []
        Int_l = []

        for i in range(n):
            e = np.random.exponential()
            tau = np.log(e * mu / np.exp(x) + 1) / mu
            t = t + tau
            T.append(t)
            x = x + mu * tau
            log_l.append(x)
            Int_l.append(e)
            x = x - alpha

        return np.array(T), np.array(log_l), np.array(Int_l)

    for _ in range(n_sequences):
        T, log_l, Int_l = self_correcting_process(1, 1, max_time * 10)
        T = T[T < max_time]
        log_l = log_l[:len(T)]
        Int_l = Int_l[:len(T)]
        score = -(log_l - Int_l).sum()

        times.append(T)
        nll.append(score)

    if filename is not None:
        mean_number_items = sum(len(t) for t in times) / len(times)
        nll = [n/mean_number_items for n in nll]
        np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)
    else:
        return times
