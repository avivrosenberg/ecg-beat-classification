import torch.nn as nn
import torch.nn.functional as nnfunc


class Encoder(nn.Module):
    def __init__(self, input_feature_size, hidden_layer_sizes):
        super(Encoder, self).__init__()

        layers = []
        prev_layer_size = input_feature_size
        for i, curr_layer_size in enumerate(hidden_layer_sizes):
            self.add_module(
                self._layer_name(i),
                nn.Sequential(
                    nn.Linear(prev_layer_size, curr_layer_size, bias=True),
                    nn.Sigmoid()
                ))
            prev_layer_size = curr_layer_size

        self.num_layers = len(hidden_layer_sizes)
        self.output_feature_size = prev_layer_size

    def forward(self, x):
        y = x
        for layer_idx in range(self.num_layers):
            layer = getattr(self, self._layer_name(layer_idx))
            y = layer(y)
        return y

    def get_layer_weights(self, layer_idx):
        if layer_idx not in range(self.num_layers):
            raise ValueError("Invalid layer index")

        layer = getattr(self, self._layer_name(layer_idx))
        layer_params = list(layer.parameters())

        # First parameter is weight, second is bias
        return layer_params[0]

    def _layer_name(self, layer_idx):
        return f'layer_{layer_idx}'


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, ext_weights=None):
        super(Decoder, self).__init__()

        self.decoder = nn.Linear(input_size, output_size, bias=True)
        self.ext_weights = ext_weights

        if self.ext_weights is not None:
            assert self.ext_weights.shape[0] == output_size
            assert self.ext_weights.shape[1] == input_size
            self.decoder.weight = None

    def forward(self, x):
        if self.ext_weights is not None:
            return nnfunc.linear(x, self.ext_weights, self.decoder.bias.data)
        else:
            return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, inout_feature_size, hidden_layer_sizes):
        super().__init__()

        self.encoder = Encoder(inout_feature_size, hidden_layer_sizes)
        self.decoder = Decoder(hidden_layer_sizes[-1], inout_feature_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderClassifier(AutoEncoder):
    def __init__(self, feature_size, hidden_layer_sizes, num_classes):
        super().__init__(feature_size, hidden_layer_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, num_classes, bias=True),
            nn.LogSoftmax(dim=1)  # 0 is the batch dimension
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

