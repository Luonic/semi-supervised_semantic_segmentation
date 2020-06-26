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


class MultiscaleFeatureDiscriminator(nn.Module):
    def __init__(self, in_channels=[48, 48 * 2, 48 * 4, 48 * 8], out_channels=[64, 128, 256, 512, 1]):
        super(MultiscaleFeatureDiscriminator, self).__init__()
        self.layers = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for i in range(1, len(out_channels) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i] + out_channels[i - 1], out_channels[i], kernel_size=4, stride=2,
                              padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.classifier = nn.Conv2d(out_channels[-2], out_channels[-1], kernel_size=1, bias=True)

    def forward(self, inputs):
        out = self.stem(inputs[0])

        for idx, layer in enumerate(self.layers):
            out = torch.cat([out, inputs[idx + 1]], dim=1)
            out = layer(out)

        out = self.classifier(out)
        return out
