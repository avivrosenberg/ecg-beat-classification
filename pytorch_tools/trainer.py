import abc
import math
import time

import torch
import torch.utils.data.dataloader as tdl


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
        return _foreach_dataloader(dl_train, self.train_batch, verbose)

    def test(self, dl_test: tdl.DataLoader, verbose=False):
        self.model.eval()
        with torch.autograd.no_grad():
            return _foreach_dataloader(dl_test, self.test_batch, verbose)

    def fit(self, dl_train, dl_test, num_epochs, verbose=False):
        results = []

        for epoch in range(num_epochs):
            if verbose:
                print(f'*** EPOCH {epoch+1}/{num_epochs} ***')

            train_loss = self.train(dl_train, verbose)
            test_loss = self.test(dl_test, verbose)

            results.append(dict(train_loss=train_loss.item(),
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


def _foreach_dataloader(loader: tdl.DataLoader, forward_fn, verbose=False):
    avg_loss = torch.tensor(0.0)
    num_batches = math.ceil(len(loader.dataset) / loader.batch_size)

    def process_item(i, d):
        return forward_fn(d)

    def process_item_verbose(i, d):
        print(f'Batch {i+1}/{num_batches}: ', end='')
        t = time.time()
        loss = forward_fn(d)
        print(f'{loss.item():.3f} ({time.time()-t:.3f} sec)')
        return loss

    process_fn = process_item_verbose if verbose else process_item

    for batch_idx, data in enumerate(loader):
        avg_loss += process_fn(batch_idx, data)

    avg_loss /= num_batches
    return avg_loss
