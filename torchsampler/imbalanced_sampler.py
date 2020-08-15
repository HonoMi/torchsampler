import torch
from torch.utils.data import Dataset as PytorchDatset


class DatasetInterface(PytorchDatset):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def build_class_balanced_sampler(dataset,
                                 get_label_func,
                                 indices=None,
                                 size=None):

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

    return WeightedDatsetSampler(dataset,
                                 weights,
                                 indices=indices,
                                 size=size)


class WeightedDatsetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,
                 dataset: DatasetInterface,
                 weights,
                 indices=None,
                 size=None):

        self.indices = indices or list(range(len(dataset)))
        self.size = size or len(self.indices)
        self.weights = torch.tensor(weights)

    def __iter__(self):
        i_s = torch.multinomial(self.weights,
                                self.size,
                                replacement=True)
        return (self.indices[i] for i in i_s)

    def __len__(self):
        return self.size
