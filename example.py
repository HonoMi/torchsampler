import torch
from torchvision.datasets.vision import VisionDataset

from torchsampler import ImbalancedDatasetSampler


def compare_sampler(dataset, label_fn):

    loaders = {
        'raw loader': torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
        ),
        'balanced loader': torch.utils.data.DataLoader(
            dataset,
            sampler=ImbalancedDatasetSampler(
                dataset,
                callback_get_label=label_fn),
            batch_size=100,
        )
    }

    for name, loader in loaders.items():
        print('\n-- {0} --'.format(name))
        class0_tot = 0
        class1_tot = 0
        for i_batch, batch in enumerate(loader):
            class0_size = len([retrieved_class_ for retrieved_class_ in batch[1]
                               if retrieved_class_ == 0])
            class1_size = len([retrieved_class_ for retrieved_class_ in batch[1]
                               if retrieved_class_ == 1])
            print(f'batch-{i_batch}:'
                  f'\tclass0 {class0_size} '
                  f'\tclass1 {class1_size} ')
            class0_tot += class0_size
            class1_tot += class1_size
        print(f'class0 total: {class0_tot}    class1 total: {class1_tot}')


if __name__ == '__main__':

    class MyDatset(VisionDataset):

        def __init__(self):
            super().__init__(None)

        def __getitem__(self, index):
            mod = index % 3
            class_ = 1 if mod - 1 >= 0 else 0
            return torch.tensor(index), class_

        def __len__(self):
            return 1000

    # Iterate native list.
    print('\n\n== Iterate native list ==')
    dataset = [
        (i, 1 if (i % 3 - 1) >= 0 else 0)
        for i in range(0, 1000)
    ]
    dataset = MyDatset()
    compare_sampler(dataset,
                    label_fn = lambda dataset, idx: dataset[idx][1])

    # Iterate custom dataset.
    print('\n\ns== Iterate custom datset ==')
    dataset = MyDatset()
    compare_sampler(dataset,
                    label_fn = lambda dataset, idx: dataset[idx][1])
