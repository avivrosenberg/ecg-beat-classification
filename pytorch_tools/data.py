import math

import numpy as np
import torch
import torchvision
from torch.utils.data import sampler as sampler


def create_train_validation_loaders(dataset, validation_ratio, batch_size,
                                    num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    num_total_samples = len(dataset)
    validation_size = math.floor(num_total_samples * validation_ratio)

    idx_samples = list(range(num_total_samples))

    # Since we're splitting at random, this is actually Monte-Carlo cross
    # validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
    idx_validation = np.random.choice(
        idx_samples, size=validation_size, replace=False
    )
    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=sampler.SubsetRandomSampler(idx_validation),
    )

    idx_train = list(set(idx_samples) - set(idx_validation))
    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=sampler.SubsetRandomSampler(idx_train),
    )

    return dl_train, dl_valid


class DatasetFolder(torchvision.datasets.DatasetFolder):
    """
    Extends the DatasetFolder form torchvision to add some useful features.
    """
    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None, ignore_classes: list=None):

        super().__init__(root, loader, extensions, transform=transform,
                         target_transform=target_transform)

        self.idx_to_class = \
            {idx: cls for cls, idx in self.class_to_idx.items()}

        self.ignore_classes = ignore_classes
        if self.ignore_classes:
            self.samples = [
                (data, class_idx) for data, class_idx in self.samples
                if self.idx_to_class[class_idx] not in ignore_classes
            ]

    def samples_per_class(self):
        samples_per_class = {cls: 0 for cls in self.classes}
        for _, class_idx in self.samples:
            samples_per_class[self.idx_to_class[class_idx]] += 1
        return samples_per_class

    def __repr__(self):
        fmt_str = super().__repr__()
        fmt_str += '\n'
        fmt_str +=\
            '    Samples per class: {}\n'.format(self.samples_per_class())
        if self.ignore_classes:
            fmt_str += \
                '    Ignored classes: {}\n'.format(self.ignore_classes)
        return fmt_str
