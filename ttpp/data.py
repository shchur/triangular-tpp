import ttpp
import numpy as np
import torch
import torch.utils.data as data_utils

from pathlib import Path
from sklearn.model_selection import train_test_split

dataset_dir = Path(__file__).parents[1] / 'data'


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.pkl'
    file_list = [x.stem for x in (dataset_dir).iterdir() if check(x)]
    return sorted(file_list)


def load_dataset(name):
    """Load dataset."""
    if not name.endswith('.pkl'):
        name += '.pkl'
    loader = torch.load(dataset_dir / name)

    sequences = loader["sequences"]
    arrival_times = [seq["arrival_times"] for seq in sequences]
    if "nll" in sequences[0].keys():
        nll = np.array([seq["nll"] for seq in sequences])
    else:
        nll = None
    t_max = loader["t_max"]
    mean_number_items = loader.get('mean_number_items')

    return SequenceDataset(arrival_times, t_max, nll=nll, mean_number_items=mean_number_items)


class SequenceDataset(data_utils.Dataset):
    def __init__(self, arrival_times, t_max, mask=None, nll=None, mean_number_items=None):
        """Dataset consisting of variable-length sequences sampled iid from the interval [0, t_max].
        
        Args:
            arrival_times: List of lists containing arrival times of events
            t_max: Length of the observed interval
            mask: Boolean mask indicating which entries in times correspond to the events (i.e. not padding)
            nll: Negative log-likelihood for each sequence (if available)
            mean_number_items: Average sequence length

        Attributes:
            times: Arrival times of events padded with t_max, shape [num_sequences, max_seq_len, 1]
            mask: Boolean mask indicating which entries in times correspond to the events (i.e. not padding)
                shape [num_sequences, max_seq_len, 1]
        """

        self.nll = nll
        self.t_max = float(t_max)
        self.mean_number_items = mean_number_items or 1.0

        if isinstance(arrival_times, torch.Tensor):
            if mask is None:
                raise ValueError('If arrival_times is a Tensor, mask must be specified')
            if not isinstance(mask, torch.Tensor):
                raise ValueError('mask must be of type torch.Tensor')
            if arrival_times.shape != mask.shape:
                raise ValueError('arrival_times should have the same shape')

            self.times = arrival_times
            self.mask = mask
        else:
            self.times = self.process_arrival_times(arrival_times)

            if self.times.min() < 0 or self.times.max() != self.t_max:
                raise ValueError("Times should be between 0 and max_iterval_time")

            self.mask = (self.times > 0).float() * (self.times != self.t_max).float()

            if self.nll is None:
                self.nll = np.zeros(len(self.times))
        self.mean_number_items = self.mask.sum([-2, -1]).mean()

    def process_arrival_times(self, arrival_times):
        max_length = max([len(x) for x in arrival_times]) + 1
        arrival_times = [np.concatenate([x, np.full(max_length - len(x), self.t_max)]) for x in arrival_times]
        arrival_times = torch.Tensor(arrival_times).type(torch.get_default_dtype())
        return arrival_times.unsqueeze(-1)

    def train_val_test_split(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=123):
        assert train_size + val_size + test_size == 1
        ind1 = int(len(self.times) * train_size)
        ind2 = int(len(self.times) * (train_size + val_size))

        dtrain = SequenceDataset(self.times[:ind1], mask=self.mask[:ind1], t_max=self.t_max,
                                 nll=self.nll[:ind1], mean_number_items=self.mean_number_items)
        dval = SequenceDataset(self.times[ind1:ind2], mask=self.mask[ind1:ind2], t_max=self.t_max,
                               nll=self.nll[ind1:ind2], mean_number_items=self.mean_number_items)
        dtest = SequenceDataset(self.times[ind2:], mask=self.mask[ind2:], t_max=self.t_max,
                                nll=self.nll[ind2:], mean_number_items=self.mean_number_items)

        return dtrain, dval, dtest

    def __getitem__(self, key):
        return self.times[key], self.mask[key]

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        return f'SequenceDataset({len(self)})'
