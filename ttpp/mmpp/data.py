import torch

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent / 'data/mmpp'


def list_datasets():
    """Returns names of all the available datasets"""
    return [f.stem for f in data_dir.iterdir()]


def save_dataset(file_name, events, states, durations, generator=None, prior=None, rates=None):
    """Save an MMPP dataset to a file."""
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    full_path = data_dir / file_name

    data_dict = {
        'events': events,
        'states': states,
        'durations': durations,
        'generator': generator,
        'prior': prior,
        'rates': rates,
    }
    torch.save(data_dict, full_path)


def load_dataset(file_name):
    """Load an MMPP dataset from a file."""
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    full_path = data_dir / file_name
    return torch.load(full_path)
