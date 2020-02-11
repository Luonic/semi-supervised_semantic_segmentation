import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, num_layers, in_channels=2, initial_channels=64, max_depth=512, out_channels=1):
        super(Discriminator, self).__init__()
        layers = nn.ModuleList()
        pred_channels = in_channels
        next_channels = initial_channels
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(pred_channels, next_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            pred_channels = next_channels
            next_channels = min(next_channels * 2, max_depth)

        layers.append(
            nn.Sequential(
                nn.Conv2d(pred_channels, out_channels, kernel_size=1, bias=False)))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
