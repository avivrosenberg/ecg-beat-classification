import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_feature_size=54, hidden_layer_sizes=(100,)):
        super(Encoder, self).__init__()

        layers = []
        prev_layer_size = input_feature_size
        for curr_layer_size in hidden_layer_sizes:
            layers += [
                nn.Linear(prev_layer_size, curr_layer_size, bias=True),
                nn.Sigmoid()
            ]
            prev_layer_size = curr_layer_size

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()

        self.decoder = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, num_classes, bias=True),
            nn.Softmax(dim=1)  # 0 is the batch dimension
        )

    def forward(self, x):
        return self.classifier(x)


