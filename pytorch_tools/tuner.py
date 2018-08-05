import abc
import json

import torch

from .data import create_train_validation_loaders
from .trainer import ModelTrainer


DEFAULT_MAX_ITER = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_CROSS_VALIDATION_K = 5


class HyperparameterTuner(abc.ABC):
    def __init__(self, max_iter=DEFAULT_MAX_ITER,
                 batch_size=DEFAULT_BATCH_SIZE,
                 cv_k=DEFAULT_CROSS_VALIDATION_K):

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.cv_k = cv_k

    @abc.abstractmethod
    def create_trainer(self, hypers: dict) -> ModelTrainer:
        """
        Creates a ModelTrainer based on a given set of hyper parameters.
        :param hypers: Hyperparameter values to create the model trainer with.
        :return: A new model trainer instance. Note: this method must return a
            new instance each time it's called.
        """
        pass

    @abc.abstractmethod
    def sample_hyperparams(self) -> dict:
        """
        :return: A dictionary containing sampled hyper parameter values.
        """
        pass

    def tune(self, ds_train, output_filename=None):
        """
        Tune model hyperparameters with k-fold cross-validation.

        :param ds_train:
        :param output_filename:
        :return:
        """
        results = []
        validation_ratio = 1 / self.cv_k

        best_hypers = None
        best_loss = None
        for i in range(self.max_iter):
            # Sample hyper params
            hypers = self.sample_hyperparams()

            print(f'# Tuning ({i+1}/{self.max_iter}): hypers={hypers}')

            avg_train_loss = torch.tensor(0.0)
            avg_valid_loss = torch.tensor(0.0)

            for k in range(self.cv_k):
                dl_train, dl_valid = create_train_validation_loaders(
                    ds_train, validation_ratio, self.batch_size
                )

                trainer = self.create_trainer(hypers)

                print(f'# Tuning ({i+1}/{self.max_iter}), k={k}, Train:')
                train_loss = trainer.train(dl_train, verbose=True)

                print(f'# Tuning ({i+1}/{self.max_iter}), k={k}, Test:')
                valid_loss = trainer.test(dl_valid, verbose=True)

                avg_train_loss += train_loss
                avg_valid_loss += valid_loss

            avg_train_loss /= self.cv_k
            avg_valid_loss /= self.cv_k
            print(f'# Tuning ({i+1}/{self.max_iter}): '
                  f'Avg. validation loss={avg_valid_loss.item():.3f}')

            if best_loss is None or avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_hypers = hypers

            results.append(dict(iter=i, hypers=hypers,
                                train_loss=avg_train_loss,
                                valid_loss=avg_valid_loss))

        if output_filename is not None:
            output_filename = f'{output_filename}.json'
            with open(output_filename, mode='w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, sort_keys=True)

        return best_hypers
