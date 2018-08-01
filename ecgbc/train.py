import itertools
import math
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data.dataloader
import torchvision.transforms
import torch.nn.functional as nnfunc

import ecgbc.dataset.transforms
import ecgbc.dataset.wfdb_dataset

import ecgbc.dataset.wfdb_single_beat
import ecgbc.models.DAE as models


DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_NOISE_STD = 0.1

tf_dataset = torchvision.transforms.Compose([
    # Not sure if normalization is required
    ecgbc.dataset.transforms.Normalize1D()
])


def train_autoencoder(ds_train, batch_size=DEFAULT_BATCH_SIZE,
                      num_epochs=DEFAULT_NUM_EPOCHS,
                      noise_std=DEFAULT_NOISE_STD,
                      target_activation=0.05,
                      reg_coeff=(1.0, .0, .0, .0),
                      **kwargs):
    """
    Trains a Denoising Autoencoder in an unsupervised fashion.
    :param ds_train:
    :param batch_size:
    :param num_epochs:
    :param noise_std:
    :param target_activation:
    :param reg_coeff:
    :param kwargs:
    :return:
    """

    train_dataset = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(
        ds_train, transform=tf_dataset
    )
    train_loader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Train", train_dataset)

    feature_size = len(train_dataset[0][0])
    hidden_layer_sizes = (100,)
    num_classes = len(train_dataset.idx_to_class.keys())

    encoder = models.Encoder(feature_size, hidden_layer_sizes)
    print(encoder)

    # Part I - Unsupervised training
    for h_idx in range(len(hidden_layer_sizes)):
        encoder_weights = encoder.get_layer_weights(h_idx).t()
        decoder = models.Decoder(hidden_layer_sizes[-1], feature_size,
                                 ext_weights=encoder_weights)

    opt_params = set(itertools.chain(encoder.parameters(),
                                     decoder.parameters()))
    optimizer = torch.optim.LBFGS(opt_params, max_iter=500)

    def loss_function(samples, encoder_outputs, decoder_outputs):
        batch_size_ratio = samples.shape[0] / len(train_dataset)

        # MSE reconstruction loss
        reconstruction_loss = 0.5 * nnfunc.mse_loss(samples, decoder_outputs)

        # L2 Parameter regularization
        regularization_loss = torch.tensor(0.0)
        for param in opt_params:
            # Don't include bias in L2 parameter regularization
            if len(param.shape) > 1:
                regularization_loss += 0.5 * torch.norm(param, p=2) ** 2
        regularization_loss *= batch_size_ratio

        # Representation sparsity loss (BCE loss of activations)
        mean_activations = encoder_outputs.mean(dim=0)
        target_activations = \
            torch.ones_like(mean_activations) * target_activation

        sparsity_loss = nnfunc.binary_cross_entropy(mean_activations,
                                                    target_activations)

        total_loss = torch.tensor(0.0)
        losses = (reconstruction_loss, regularization_loss, sparsity_loss)
        for i, loss in enumerate(losses):
            total_loss += reg_coeff[i] * loss

        return total_loss

    num_batches_train = math.ceil(len(train_dataset) / batch_size)
    for epoch_idx in range(num_epochs):
        print(f'*** Starting epoch {epoch_idx+1}/{num_epochs} ***')

        for batch_idx, (samples, _) in enumerate(train_loader):
            print(f'>>> Batch {batch_idx+1}/{num_batches_train}: ', end='')

            def lbfgs_step():
                samples_noised = torch.randn_like(samples)*noise_std + samples
                samples_encoded = encoder(samples_noised)
                samples_decoded = decoder(samples_encoded)

                loss = loss_function(samples, samples_encoded, samples_decoded)

                optimizer.zero_grad()
                loss.backward()
                return loss

            t = time.time()
            training_loss = optimizer.step(lbfgs_step)
            print(f'loss={training_loss:.3f} ({time.time()-t:.3f}s)')


def train_classifier():
    pass


def test(ds_test, batch_size=DEFAULT_BATCH_SIZE):
    test_dataset = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(
        ds_test, transform=tf_dataset
    )
    test_loader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Test", test_dataset)

    # Print test-set accuracy
    num_batches_test = math.ceil(len(test_dataset) / batch_size)
    test_loss = torch.tensor(0.0)
    for batch_idx, (samples, _) in enumerate(test_loader):
        # TODO
        pass

    print(f'>>> Test loss={test_loss.item():.3f}')
