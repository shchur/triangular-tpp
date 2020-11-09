import numpy as np
from pathlib import Path
dataset_dir = Path(__file__).parents[2] / 'data'

def generate(max_time, n_sequences, filename='nonstationary_poisson'):
    amp = 0.99
    l_t = lambda t: np.sin(2 * np.pi * t / max_time) * amp + 1
    l_int = lambda t1, t2, L: -L / (2 * np.pi) * (np.cos(2 * np.pi * t2 / L) - np.cos(2 * np.pi * t1 / L)) * amp + (t2 - t1)

    times, nll = [], []

    for _ in range(n_sequences):
        while 1:
            T = np.random.exponential(size=max_time * 3).cumsum() * 0.5
            r = np.random.rand(max_time * 3)
            index = r < l_t(T) / 2.0

            T = T[index]

            if T.max() > max_time:
                T = T[T < max_time]
                score = -(np.log(l_t(T)).sum() - l_int(T[0], T[-1], len(T)))
                break
        times.append(T)
        nll.append(score)

    if filename is not None:
        mean_number_items = sum(len(t) for t in times) / len(times)
        nll = [n/mean_number_items for n in nll]
        np.savez(f'{dataset_dir}/{filename}.npz', arrival_times=times, nll=nll, t_max=max_time, mean_number_items=mean_number_items)
    else:
        return times
