import numpy as np
from pathlib import Path
dataset_dir = Path(__file__).parents[2] / 'data'

def generate(max_time, n_sequences, filename='stationary_poisson'):
    N = int(max_time) * 5

    times, nll = [], []
    for _ in range(n_sequences):
        T = np.random.exponential(size=N)
        T = T.cumsum()

        if T.max() < max_time:
            raise ValueError('Max of sequence lower than max_time')

        T = T[T < max_time]
        times.append(T)
        nll.append(1)

    if filename is not None:
        mean_number_items = sum(len(t) for t in times) / len(times)
        nll = [n/mean_number_items for n in nll]
        np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)
    else:
        return times
