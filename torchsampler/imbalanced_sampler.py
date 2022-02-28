from typing import Optional

import torch
from torch.utils.data import Dataset as PytorchDatset
import logging

logger = logging.getLogger(__name__)


class DatasetInterface(PytorchDatset):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def build_class_balanced_sampler(dataset,
                                 get_label_func,
                                 indices=None,
                                 **kwargs):

    indices = indices or list(range(len(dataset)))
    label_to_count = {}
    for idx in indices:
        label = get_label_func(dataset, idx)
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1

    # weight for each sample
    weights = [1.0 / label_to_count[get_label_func(dataset, idx)]
               for idx in indices]
    weights = torch.DoubleTensor(weights)

    return WeightedDatasetSampler(len(dataset),
                                  sample_weights=weights,
                                  dataset_indices=indices,
                                  **kwargs)


class WeightedDatasetSampler(torch.utils.data.sampler.Sampler):
    """WeightedDatasetSampler

    sample_weights=None で呼んだ場合，サンプリングは行われず，単にデータセット中のサンプル全てが使われる．
    一様分布からサンプリングされる，という訳ではないことに注意．
    """

    # TODO: fix_seed => fix_seed_for_weighted_sampler
    def __init__(self,
                 # dataset: DatasetInterface,
                 dataset_size: int,
                 sample_weights: Optional[torch.Tensor],
                 same_samples_over_epochs=False,
                 sample_then_shuffle_every_epoch=False,
                 dataset_indices=None,
                 seed_for_sampling: Optional[int] = None):

        # self._dataset_indices = dataset_indices or list(range(len(dataset)))
        self._size = dataset_size
        self._dataset_indices = dataset_indices\
            or torch.tensor(list(range(dataset_size)), dtype=torch.int64, requires_grad=False)

        if sample_weights is None:
            self._sample_weights = None
            # self._sample_weights = torch.tensor([1 / self._size] * self._size,
            #                                     dtype=torch.float64,
            #                                     requires_grad=False)
        else:
            if len(sample_weights) != self._size:
                raise ValueError('len(weights) != datase size')
            self._sample_weights = torch.tensor(sample_weights, requires_grad=False)
            self._sample_weights = self._sample_weights / self._sample_weights.sum()
        self._same_samples_over_epochs = same_samples_over_epochs
        self._sample_then_shuffle = sample_then_shuffle_every_epoch
        self._cache = None
        self._seed_for_sampling = seed_for_sampling

    def __iter__(self):
        if self._same_samples_over_epochs and self._cache is not None:
            # print('\n\n == WeightedDatasetSampler ==')
            # print(len(self._cache))
            # print(self._cache)
            return iter(self._cache)

        with torch.no_grad():

            if self._sample_weights is None:
                indices = self._dataset_indices
            else:
                if self._seed_for_sampling:
                    rand_state = torch.random.get_rng_state()

                    logger.info('setting seed to "%d" temporarily for weighted sampling',
                                self._seed_for_sampling)
                    torch.manual_seed(self._seed_for_sampling)
                    torch.cuda.manual_seed(self._seed_for_sampling)
                    indices = torch.multinomial(self._sample_weights,
                                                self._size,
                                                replacement=True)
                    torch.random.set_rng_state(rand_state)
                else:
                    indices = torch.multinomial(self._sample_weights,
                                                self._size,
                                                replacement=True)

            if self._sample_then_shuffle:
                indices = indices[torch.randperm(len(indices))]

            datsaet_indices = [self._dataset_indices[idx] for idx in indices]

            if self._same_samples_over_epochs:
                self._cache = datsaet_indices
            # print('\n\n == WeightedDatasetSampler ==')
            # print(len(datsaet_indices))
            # print(datsaet_indices)

            return iter(datsaet_indices)

    def __len__(self):
        return self._size
