import abc
import math
import time

import torch
import torch.utils.data.dataloader as tdl
import tqdm


class ModelTrainer(abc.ABC):
    """
    Class containing boilerplate code for training pytorch models.
    A trainer consists of a Model, a Loss function, an Optimizer and a
    set of Hyperparameter values.
    """

    def __init__(self, **hyperparams):
        # Set hyperparams as attributes of the trainer
        for hp in hyperparams:
            setattr(self, hp, hyperparams[hp])

        # Construct the modules
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()
        self.loss_fn = self.create_loss()

    def train(self, dl_train: tdl.DataLoader, verbose=False):
        self.model.train()
        progress_bar = 'Train' if verbose else None
        return _foreach_dataloader(dl_train, self.train_batch, progress_bar)

    def test(self, dl_test: tdl.DataLoader, verbose=False):
        self.model.eval()
        progress_bar = 'Test' if verbose else None
        with torch.autograd.no_grad():
            return _foreach_dataloader(dl_test, self.test_batch, progress_bar)

    def fit(self, dl_train, dl_test, num_epochs, verbose=False):
        results = []

        for epoch in range(num_epochs):
            if verbose:
                print(f'*** EPOCH {epoch+1}/{num_epochs} ***')

            train_loss = self.train(dl_train, verbose)
            test_loss = self.test(dl_test, verbose)

            results.append(dict(epoch=epoch,
                                train_loss=train_loss.item(),
                                test_loss=test_loss.item()))
        return results

    @abc.abstractmethod
    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    @abc.abstractmethod
    def create_loss(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(self, dl_sample) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, dl_sample) -> torch.Tensor:
        raise NotImplementedError()


def _foreach_dataloader(loader: tdl.DataLoader, forward_fn, progress=None):
    avg_loss = torch.tensor(0.0)
    num_batches = math.ceil(len(loader.dataset) / loader.batch_size)

    pbar = None

    def process_item(i, d):
        return forward_fn(d)

    def process_item_with_pbar(i, d):
        loss = forward_fn(d)
        pbar.set_description(f'{progress} ({loss.item():.3f})')
        pbar.update(i)
        return loss

    if progress is not None:
        total = math.ceil(len(loader.dataset) / loader.batch_size)
        pbar = tqdm.tqdm(desc=progress, total=total)
        process_fn = process_item_with_pbar
    else:
        process_fn = process_item

    for batch_idx, data in enumerate(loader):
        avg_loss += process_fn(batch_idx, data)

    avg_loss /= num_batches

    if pbar:
        pbar.set_description(f'{progress} (Avg. {avg_loss.item():.3f})')

    return avg_loss
