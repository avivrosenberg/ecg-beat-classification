import abc
import math
import time

import torch
import torch.utils.data.dataloader


class ModelTrainer(abc.ABC):
    """
    Class containing boilerplate code for training pytorch models.
    A trainer consists of a Model, and an Optimizer and a set of Hyperparameter
    values.
    """

    def __init__(self):
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, dl_train: torch.utils.data.dataloader.DataLoader,
              verbose=False):
        self.model.train()
        return _foreach_dataloader(dl_train, self.train_batch, verbose)

    def test(self, dl_test: torch.utils.data.dataloader.DataLoader,
             verbose=False):
        self.model.eval()
        with torch.autograd.no_grad():
            return _foreach_dataloader(dl_test, self.test_batch, verbose)

    @abc.abstractmethod
    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(self, dl_sample) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, dl_sample) -> torch.Tensor:
        raise NotImplementedError()


def _foreach_dataloader(loader: torch.utils.data.dataloader.DataLoader,
                        forward_fn, verbose=False):
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
