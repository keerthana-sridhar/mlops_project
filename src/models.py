# src/models.py

import torch
import torch.nn as nn
from torchvision import models


class ConfigCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        channels = config["conv_channels"]
        k = config["kernel_size"]

        layers = []
        in_c = 3

        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, k))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_c = out_c

        self.conv = nn.Sequential(*layers)

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, config["fc_dim"]),
            nn.ReLU(),
            nn.Linear(config["fc_dim"], 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_model(model_name, params):
    if model_name == "cnn":
        return ConfigCNN(params["model"]["cnn"])

    elif model_name == "resnet":
        model = models.resnet18(pretrained=params["model"]["resnet"]["pretrained"])
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model