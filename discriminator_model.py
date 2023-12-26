import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=3, stride=1, padding=1, padding_mode="reflect", groups=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers = []
        in_channels = features[0]
        for idx, feature in enumerate(features[1:]):
            layers.append(CNNBlock(in_channels, feature, stride=1 if idx == len(features) - 1 else 2))
            in_channels = feature

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1, padding_mode="reflect"),
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        print("Input size:", x.size())
        x = self.model(self.initial(x))
        print("Output size:", x.size())
        return x
