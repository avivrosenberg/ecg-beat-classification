import numpy as np
import torch
import torch.utils.data

import ecgbc.models as models
import ecgbc.losses as losses

import pytorch_tools.trainer as trainer
import pytorch_tools.tuner as tuner

DEFAULT_FEATURE_SIZE = 54
DEFAULT_HIDDEN_LAYER_SIZES = (100,)
DEFAULT_NOISE_STD = 0.1
DEFAULT_TARGET_ACTIVATION = 0.05
DEFAULT_REGULARIZATION_COEFFICIENTS = (.5, .5)


class Trainer(trainer.ModelTrainer):
    def __init__(self,
                 feature_size=DEFAULT_FEATURE_SIZE,
                 hidden_layer_sizes=DEFAULT_HIDDEN_LAYER_SIZES,
                 noise_std=DEFAULT_NOISE_STD,
                 target_activation=DEFAULT_TARGET_ACTIVATION,
                 reg_coeff=DEFAULT_REGULARIZATION_COEFFICIENTS,
                 **kwargs):

        # Hyperparameters
        self.feature_size = feature_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.noise_std = noise_std
        self.target_activation = target_activation
        self.reg_coeff = reg_coeff

        # Init model and optimizer
        super().__init__(**kwargs)

    def create_model(self) -> torch.nn.Module:
        return models.AutoEncoder(self.feature_size, self.hidden_layer_sizes)

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.model.parameters(), max_iter=500)

    def create_loss(self) -> torch.nn.Module:
        return losses.DAESparseL2Loss(
            l2_regularization_params=self.model.parameters(),
            l2_regularization_coeff=self.reg_coeff[0],
            sparsity_loss_target_activation=self.target_activation,
            sparsity_loss_coeff=self.reg_coeff[1]
        )

    def train_batch(self, dl_batch) -> trainer.BatchResult:
        samples, _ = dl_batch

        def lbfgs_step():
            samples_noised = \
                torch.randn_like(samples) * self.noise_std + samples

            samples_encoded = self.model.encoder(samples_noised)
            samples_decoded = self.model.decoder(samples_encoded)

            loss = self.loss_fn(samples, samples_encoded, samples_decoded)

            self.optimizer.zero_grad()
            loss.backward()
            return loss

        final_loss = self.optimizer.step(lbfgs_step)
        return trainer.BatchResult(loss=final_loss.item(), num_correct=0)

    def test_batch(self, dl_batch) -> trainer.BatchResult:
        samples, _ = dl_batch

        samples_noised = \
            torch.randn_like(samples) * self.noise_std + samples

        samples_encoded = self.model.encoder(samples_noised)
        samples_decoded = self.model.decoder(samples_encoded)

        loss = self.loss_fn(samples, samples_encoded, samples_decoded)
        return trainer.BatchResult(loss=loss, num_correct=0)


class Tuner(tuner.HyperparameterTuner):
    def create_trainer(self, hypers) -> trainer.ModelTrainer:
        return Trainer(**hypers)

    def sample_hyperparams(self):
        return dict(
            feature_size=DEFAULT_FEATURE_SIZE,
            hidden_layer_sizes=DEFAULT_HIDDEN_LAYER_SIZES,
            target_activation=DEFAULT_TARGET_ACTIVATION,
            reg_coeff=(
                10 ** np.random.uniform(-2, 0),
                10 ** np.random.uniform(-2, 0),
            ),
        )
