import torch
import torch.nn.functional as nnfunc


class DAESparseL2Loss(torch.nn.Module):
    """
    Loss function for a DAE (denoising autoencoder).
    Implements reconstruction loss, sparsity penalty on the encoding
    (hidden layer activations) and L2 regularization of the model parameters.
    """
    def __init__(self,
                 l2_regularization_params,
                 l2_regularization_coeff,
                 sparsity_loss_target_activation,
                 sparsity_loss_coeff):

        super().__init__()
        self.l2_regularization_params = l2_regularization_params
        self.l2_regularization_coeff = l2_regularization_coeff
        self.sparsity_loss_target_activation = sparsity_loss_target_activation
        self.sparsity_loss_coeff = sparsity_loss_coeff

    def forward(self, samples, encoder_outputs, decoder_outputs):
        # MSE reconstruction loss
        reconstruction_loss = 0.5 * nnfunc.mse_loss(samples, decoder_outputs)

        # L2 Parameter regularization
        regularization_loss = torch.tensor(0.0)
        for param in self.l2_regularization_params:
            # Don't include bias in L2 parameter regularization
            if len(param.shape) > 1:
                regularization_loss += 0.5 * torch.norm(param, p=2) ** 2

        # Representation sparsity loss (BCE loss of activations)
        mean_activations = encoder_outputs.mean(dim=0)
        target_activations = \
            torch.ones_like(mean_activations) * \
            self.sparsity_loss_target_activation

        sparsity_loss = nnfunc.binary_cross_entropy(mean_activations,
                                                    target_activations)

        total_loss = torch.tensor(0.0)
        coeffs = (1, self.l2_regularization_coeff, self.sparsity_loss_coeff)
        losses = (reconstruction_loss, regularization_loss, sparsity_loss)
        for i, loss in enumerate(losses):
            total_loss += coeffs[i] * loss

        return total_loss

