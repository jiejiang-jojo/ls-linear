"""The module for data loading."""

import logging

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from utils.flags import with_flags


class ChunkDataset(Dataset):
    def __getitem__(self, idx):
        chunk = self._getitemchunk(idx)
        return chunk[0][0], chunk[1][0]

    def _getitemchunk(self, idx, chunk_size=1):
        raise NotImplementedError


@with_flags
class SplittedDataset(ChunkDataset):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--splitted_dataset_ratio', type=str, default='7:1:2', help='The ratio for splitting the dataset into train/validation/test.')

    def __init__(self, dataset, split_name='test'):
        """Splitting data in train, validation and test data."""
        assert len(self.args.splitted_dataset_ratio.split(':')) == 3
        self.dataset = dataset
        self.split_name = split_name
        self.split_points = SplittedDataset._compute_split_points(self.args.splitted_dataset_ratio, len(dataset))
        self.name_mapping = {
                "train": 0,
                "validation": 1,
                "test": 2
                }

    @staticmethod
    def _compute_split_points(ratio: str, dlen: int):
        ps = np.array([0] + [float(x) for x in ratio.split(':')]).cumsum()
        ps = (ps / ps.max() * dlen).astype('int')
        return list(zip(ps, ps[1:]))

    def __len__(self):
        split = self.split_points[self.name_mapping[self.split_name]]
        return split[1] - split[0]

    def _getitemchunk(self, idx, chunk_size=1):
        split = self.split_points[self.name_mapping[self.split_name]]
        return self.dataset._getitemchunk(split[0] + idx, chunk_size)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def input_shape(self):
        return self.dataset.input_shape()

    def output_shape(self):
        return self.dataset.output_shape()


@with_flags
class SlideWindowDataset(ChunkDataset):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--slide_window_len', default=120, type=int, help='The input window len for sliding window on input sequence.')

    def __init__(self, dataset, output_win=1, win_offset_from_end=-1):
        self.dataset = dataset
        self.input_win = self.args.slide_window_len
        self.output_win = output_win
        self.output_win_start = self.input_win + win_offset_from_end
        self.len = len(dataset) - (self.output_win_start + self.output_win) + 1

    def __len__(self):
        return self.len

    def _getitemchunk(self, idx, chunk_size=1):
        load_chunk_size = chunk_size + (self.output_win_start + self.output_win) - 1
        x, y = self.dataset._getitemchunk(idx, load_chunk_size)
        x = x[:self.input_win + chunk_size - 1]
        y = y[self.output_win_start:self.output_win_start + self.output_win + chunk_size - 1]
        win_chunk_x = as_strided(x, (chunk_size,) + x.shape, x.strides[:1] + x.strides)
        win_chunk_y = as_strided(y, (chunk_size,) + y.shape, y.strides[:1] + y.strides)

        return win_chunk_x, win_chunk_y

    def input_shape(self):
        return (self.input_win,) + self.dataset.input_shape()

    def output_shape(self):
        return (self.output_win,) + self.dataset.output_shape()


@with_flags
class GasWellDataset(ChunkDataset):
    """Assuming the format [ts, target, inputs...]"""
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--gas_well_data_path', type=str, default='data/A7_imputed.csv', help='Path to the gas data.')

    def __init__(self, path=None):
        df = pd.read_csv(self.args.gas_well_data_path if path is None else path)
        self.data = df.values[:, 1:].astype('float32')
        logging.info(f'mean: {np.mean(self.data[:int(0.7*len(self.data))], axis=0)}')
        logging.info(f'std: {np.std(self.data[:int(0.7*len(self.data))], axis=0)}')
        self.data = (self.data - np.mean(self.data[:int(0.7*len(self.data))], axis=0)) / np.std(self.data[:int(0.7*len(self.data))], axis=0)

    def __len__(self):
        return self.data.shape[0]

    def _getitemchunk(self, idx, chunk_size=1):
        rows = self.data[idx:idx + chunk_size]
        return rows[:, 1:], rows[:, 0:1]

    def input_shape(self):
        return (self.data.shape[1] - 1,)

    def output_shape(self):
        return (1,)
