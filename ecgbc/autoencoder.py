import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as nnfunc

import ecgbc.models.DAE as models

import pytorch_tools.trainer as trainer
import pytorch_tools.tuner as tuner
from pytorch_tools.trainer import ModelTrainer

DEFAULT_NOISE_STD = 0.1


class Trainer(trainer.ModelTrainer):

    def __init__(self,
                 feature_size=54,
                 hidden_layer_sizes=(100,),
                 noise_std=DEFAULT_NOISE_STD,
                 target_activation=0.05,
                 reg_coeff=(.5, .5),
                 ):

        # Hyperparameters
        self.feature_size = feature_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.noise_std = noise_std
        self.target_activation = target_activation
        self.reg_coeff = reg_coeff

        # Init model and optimizer
        super(Trainer, self).__init__()

    def create_model(self) -> torch.nn.Module:
        return models.AutoEncoder(self.feature_size, self.hidden_layer_sizes)

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.model.parameters(), max_iter=500)

    def train_batch(self, dl_sample) -> torch.Tensor:
        samples, _ = dl_sample

        def lbfgs_step():
            samples_noised = \
                torch.randn_like(samples) * self.noise_std + samples

            samples_encoded = self.model.encoder(samples_noised)
            samples_decoded = self.model.decoder(samples_encoded)

            loss = self.loss_function(samples, samples_encoded,
                                      samples_decoded)

            self.optimizer.zero_grad()
            loss.backward()
            return loss

        return self.optimizer.step(lbfgs_step)

    def test_batch(self, dl_sample) -> torch.Tensor:
        samples, _ = dl_sample

        samples_noised = \
            torch.randn_like(samples) * self.noise_std + samples

        samples_encoded = self.model.encoder(samples_noised)
        samples_decoded = self.model.decoder(samples_encoded)

        return self.loss_function(samples, samples_encoded, samples_decoded)

    def loss_function(self, samples, encoder_outputs, decoder_outputs):
        # MSE reconstruction loss
        reconstruction_loss = 0.5 * nnfunc.mse_loss(samples, decoder_outputs)

        # L2 Parameter regularization
        regularization_loss = torch.tensor(0.0)
        for param in self.model.parameters():
            # Don't include bias in L2 parameter regularization
            if len(param.shape) > 1:
                regularization_loss += 0.5 * torch.norm(param, p=2) ** 2

        # Representation sparsity loss (BCE loss of activations)
        mean_activations = encoder_outputs.mean(dim=0)
        target_activations = \
            torch.ones_like(mean_activations) * self.target_activation

        sparsity_loss = nnfunc.binary_cross_entropy(mean_activations,
                                                    target_activations)

        total_loss = torch.tensor(0.0)
        loss_coeff = tuple([1.0, *self.reg_coeff])
        losses = (reconstruction_loss, regularization_loss, sparsity_loss)
        for i, loss in enumerate(losses):
            total_loss += loss_coeff[i] * loss

        return total_loss


class Tuner(tuner.HyperparameterTuner):
    def create_trainer(self, hypers) -> ModelTrainer:
        return Trainer(**hypers)

    def sample_hyperparams(self):
        return dict(
            feature_size=54,
            hidden_layer_sizes=(100,),
            target_activation=0.05,
            reg_coeff=(
                10**np.random.uniform(-2, 0),
                10**np.random.uniform(-2, 0),
            ),
        )

