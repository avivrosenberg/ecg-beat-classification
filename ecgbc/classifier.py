import numpy as np
import torch
import torch.utils.data

import ecgbc.models as models

import pytorch_tools.trainer as trainer
import pytorch_tools.tuner as tuner


DEFAULT_FEATURE_SIZE = 54
DEFAULT_HIDDEN_LAYER_SIZES = (100,)
DEFAULT_NUM_CLASSES = 5
DEFAULT_LEARN_RATE = 1
DEFAULT_MOMENTUM = 0.5
DEFAULT_WEIGHT_DECAY = 0.1


class Trainer(trainer.ModelTrainer):
    def __init__(self,
                 load_params_file=None,
                 feature_size=DEFAULT_FEATURE_SIZE,
                 hidden_layer_sizes=DEFAULT_HIDDEN_LAYER_SIZES,
                 num_classes=DEFAULT_NUM_CLASSES,
                 learn_rate=DEFAULT_LEARN_RATE,
                 momentum=DEFAULT_MOMENTUM,
                 weight_decay=DEFAULT_WEIGHT_DECAY,
                 **kwargs):

        # Hyperparams
        self.load_params_file = load_params_file
        self.feature_size = feature_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_classes = num_classes
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Model & Optimizer
        super().__init__(**kwargs)

    def create_model(self) -> torch.nn.Module:
        model = models.AutoEncoderClassifier(self.feature_size,
                                             self.hidden_layer_sizes,
                                             self.num_classes)

        if self.load_params_file is not None:
            loaded_state_dict = torch.load(self.load_params_file)
            model.load_state_dict(loaded_state_dict, strict=False)

        return model

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(),
                               lr=self.learn_rate,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)

    def create_loss(self) -> torch.nn.Module:
        return torch.nn.NLLLoss()

    def train_batch(self, dl_sample) -> torch.Tensor:
        (samples, targets) = dl_sample

        predicted_labels = self.model(samples)

        loss = self.loss_fn(predicted_labels, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test_batch(self, dl_sample) -> torch.Tensor:
        (samples, targets) = dl_sample
        predicted_labels = self.model(samples)
        loss = self.loss_fn(predicted_labels, targets)
        return loss


class Tuner(tuner.HyperparameterTuner):
    def sample_hyperparams(self) -> dict:
        return dict(
            feature_size=DEFAULT_FEATURE_SIZE,
            hidden_layer_sizes=DEFAULT_HIDDEN_LAYER_SIZES,
            num_classes=DEFAULT_NUM_CLASSES,
            learn_rate=10 ** np.random.uniform(-0.3, 0.1),
            momentum=10 ** np.random.uniform(-0.4, 0),
            weight_decay=10 ** np.random.uniform(-1.1, -0.3),
        )

    def create_trainer(self, hypers: dict) -> trainer.ModelTrainer:
        return Trainer(**hypers)
