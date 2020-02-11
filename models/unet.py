import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_in_channels, out_channels, shrink=True, norm_layer=nn.BatchNorm2d, train_upsampling=False):
        super(UpBlock, self).__init__()
        if train_upsampling:
            self.upsampler = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU()
            )

        self.conv3_0 = ConvBlock(out_channels + skip_in_channels, out_channels, 3, norm_layer=norm_layer)
        if shrink:
            self.conv3_1 = ConvBlock(out_channels, out_channels // 2, 3, norm_layer=norm_layer)
        else:
            self.conv3_1 = ConvBlock(out_channels, out_channels, 3, norm_layer=norm_layer)

    def forward(self, x, skip):
        x = self.upsampler(x)

        if skip.size(2) > x.size(2):
            skip = _center_crop(skip, x.size()[2:])
        elif skip.size(2) < x.size(2):
            x = _center_crop(x, skip.size()[2:])

        x = torch.cat((x, skip), dim=1)

        x = self.conv3_0(x)
        x = self.conv3_1(x)

        return x

def _center_crop(tensor, target_size):
    _, _, tensor_h, tensor_w = tensor.size()
    diff_h = (tensor_h - target_size[0])
    diff_w = (tensor_w - target_size[1])

    from_h, from_w = diff_h // 2, diff_w // 2
    to_h = target_size[0] + from_h
    to_w = target_size[1] + from_w
    return tensor[:, :, from_h: to_h, from_w: to_w]


class UNet(nn.Module):
    def __init__(self, num_classes, encoder, max_width, norm_layer=nn.BatchNorm2d, train_upsampling=False):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.decoder = nn.ModuleList()
        self.num_classes = num_classes
        prev_ch = encoder.endpoint_depths[-1]
        for idx in reversed(range(len(encoder.endpoints) - 1)):
            skip_ch = encoder.endpoint_depths[idx]
            block_width = min(encoder.endpoint_depths) * (2 ** idx)
            block_ch = min(block_width, max_width)
            block_shrink = False


            self.decoder.append(
                UpBlock(prev_ch, skip_ch, block_ch, shrink=block_shrink, norm_layer=norm_layer,
                        train_upsampling=train_upsampling)
            )
            # print(f'Decoder layer: out {block_ch}')

            prev_ch = block_ch // 2 if block_shrink else block_ch

        self.final_block = nn.Conv2d(prev_ch, num_classes, 1, bias=False)

    def forward(self, x):
        encoder_output = self.encoder(x)

        x = encoder_output[-1]
        for idx, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, encoder_output[-idx - 2])

        x = self.final_block(x)
        return x

    def get_params_with_layerwise_lr(self, encoder_lr, decoder_lr, classifier_lr):
        params = []
        params.append({'params': self.encoder.parameters(), 'lr': encoder_lr})
        params.append({'params': self.decoder.parameters(), 'lr': decoder_lr})
        params.append({'params': self.final_block.parameters(), 'lr': classifier_lr})
        return params