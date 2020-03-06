from torch.utils.data import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler


class _BaseDataLoaderIter(object):

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        # data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if isinstance(index, list):
            template_instance = self._dataset[index[0]]
            if isinstance(template_instance, (list, tuple)):  # ここは，
                length = len(list(template_instance))
                ret_data = []
                for i in range(0, length):
                    ret_data.append([list(self._dataset[idx])[i] for idx in index])
                return ret_data
            else:
                return [self._dataset[idx] for idx in index]
        else:
            return self._dataset[index]

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self._index_sampler)


class DataLoader(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, drop_last=False):

        self.dataset = dataset

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return _BaseDataLoaderIter(self)

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler
