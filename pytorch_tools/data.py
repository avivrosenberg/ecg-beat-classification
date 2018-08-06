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
                 target_transform=None):

        super().__init__(root, loader, extensions, transform=transform,
                         target_transform=target_transform)

        self.idx_to_class = \
            {idx: cls for cls, idx in self.class_to_idx.items()}

    def samples_per_class(self):
        samples_per_class = {cls: 0 for cls in self.classes}
        for _, class_idx in self.samples:
            samples_per_class[self.idx_to_class[class_idx]] += 1
        return samples_per_class

    def __repr__(self):
        fmt_str = super().__repr__()
        fmt_str += '\n'
        fmt_str += f'    Samples per class: {self.samples_per_class()}'
        return fmt_str


class SubsetDatasetFolder(DatasetFolder):
    """
    A dataset folder that only loads a subset of the data in the given
    folder. Useful for ignoring classes or taking a small portion of a dataset.
    """
    def __init__(self, root, loader, extensions, subset: dict=None,
                 transform=None, target_transform=None, random_order=False):

        super().__init__(root, loader, extensions, transform=transform,
                         target_transform=target_transform)

        if subset is None:
            self.subset = None
            return

        self.subset = dict(subset)

        samples_idx = list(range(len(self.samples)))
        if random_order:
            samples_idx = np.random.permutation(samples_idx)

        for class_label in self.classes:
            if class_label not in self.subset:
                self.subset[class_label] = math.inf

        # Create new class labels and mapping for classes that have at least
        # one sample
        new_classes = [label for label, num in self.subset.items() if num > 0]
        new_class_to_idx = {cls: idx for idx, cls in enumerate(new_classes)}
        new_idx_to_class = {idx: cls for cls, idx in new_class_to_idx.items()}

        # Take requested subset of the data and change class indices
        new_samples = []
        for idx in samples_idx:
            (path, class_idx) = self.samples[idx]
            class_label = self.idx_to_class[class_idx]
            if self.subset[class_label] > 0:
                new_class_idx = new_class_to_idx[class_label]
                new_samples.append((path, new_class_idx))
                self.subset[class_label] -= 1

        # Replace old with new
        self.samples = new_samples
        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        self.idx_to_class = new_idx_to_class

    def __repr__(self):
        fmt_str = super().__repr__()
        fmt_str += '\n'
        if self.subset is not None:
            subset_disp = {k: v for k, v in self.subset.items()
                           if not math.isinf(v)}
            fmt_str += f'    Subset loaded per class: {subset_disp}'
        return fmt_str
