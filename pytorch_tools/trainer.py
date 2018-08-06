import abc
import math

import torch
import torch.utils.data.dataloader as tdl
import tqdm

from typing import Callable, Any, NamedTuple, List, Dict


class BatchResult(NamedTuple):
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    avg_loss: float
    accuracy: float


class FitResult(NamedTuple):
    num_epochs: int
    hyperparams: Dict[str, Any]
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


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

    def train(self, dl_train: tdl.DataLoader, verbose=False) -> EpochResult:
        self.model.train()
        progress_bar = 'Train' if verbose else None
        return _foreach_batch(dl_train, self.train_batch, progress_bar)

    def test(self, dl_test: tdl.DataLoader, verbose=False) -> EpochResult:
        self.model.eval()
        progress_bar = 'Test' if verbose else None
        with torch.autograd.no_grad():
            return _foreach_batch(dl_test, self.test_batch, progress_bar)

    def fit(self, dl_train, dl_test, num_epochs, verbose=False) -> FitResult:

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        result = FitResult(num_epochs, self.hyperparams(), train_loss,
                           train_acc, test_loss, test_acc)

        for epoch in range(num_epochs):
            if verbose:
                print(f'*** EPOCH {epoch+1}/{num_epochs} ***')

            train_result = self.train(dl_train, verbose)
            test_result = self.test(dl_test, verbose)

            train_loss.append(train_result.avg_loss)
            train_acc.append(train_result.accuracy)
            test_loss.append(test_result.avg_loss)
            test_acc.append(test_result.accuracy)

        return result

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
    def train_batch(self, dl_batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param dl_batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, dl_batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param dl_batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    def hyperparams(self):
        hypers = {}

        non_hyper_attrs = ('model', 'optimizer', 'loss_fn')
        for attr in dir(self):
            if attr.startswith('_') or attr in non_hyper_attrs:
                continue
            attr_value = getattr(self, attr)
            if callable(attr_value):
                continue
            hypers[attr] = attr_value
        return hypers


def _foreach_batch(loader: tdl.DataLoader,
                   forward_fn: Callable[[Any], BatchResult],
                   progress: str = None) -> EpochResult:
    pbar = None
    total_loss = 0.0
    num_correct = 0

    num_samples = len(loader.sampler)
    num_batches = len(loader.batch_sampler)

    def process_item(i, d):
        return forward_fn(d)

    def process_item_with_pbar(i, d):
        bres = forward_fn(d)
        pbar.set_description(f'{progress} ({bres.loss:.3f})')
        pbar.update()
        return bres

    if progress is not None:
        pbar = tqdm.tqdm(desc=progress, total=num_batches)
        process_fn = process_item_with_pbar
    else:
        process_fn = process_item

    for batch_idx, data in enumerate(loader):
        batch_result = process_fn(batch_idx, data)
        total_loss += batch_result.loss
        num_correct += batch_result.num_correct

    avg_loss = total_loss / num_batches
    accuracy = 100. * num_correct / num_samples

    if pbar:
        pbar.set_description(f'{progress} '
                             f'(Avg. Loss {avg_loss:.3f}, '
                             f'Accuracy {accuracy:.1f})')

    return EpochResult(avg_loss=avg_loss, accuracy=accuracy)
