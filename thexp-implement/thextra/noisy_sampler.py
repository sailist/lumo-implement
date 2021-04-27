from typing import Iterator, Sized
from itertools import chain
import numpy as np

from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler


class NoisySampler(Sampler):
    def __init__(self, data_source: Sized, clean_mask) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.clean_ids = np.where(clean_mask)[0]
        self.noisy_ids = np.where(np.logical_not(clean_mask))[0]

    def __iter__(self):
        np.random.shuffle(self.clean_ids)
        np.random.shuffle(self.noisy_ids)
        return chain(self.clean_ids, self.noisy_ids)

    def __len__(self) -> int:
        return len(self.data_source)
