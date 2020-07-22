# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
import os

import torch
import torch.nn as nn

try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

from typing import List, Tuple

from yacs.config import CfgNode as CN

# large net
# POSE_HIGHER_RESOLUTION_NET = CN()
# POSE_HIGHER_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
# POSE_HIGHER_RESOLUTION_NET.STEM_INPLANES = 64
# POSE_HIGHER_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
# POSE_HIGHER_RESOLUTION_NET.NUM_JOINTS = 2
# POSE_HIGHER_RESOLUTION_NET.TAG_PER_JOINT = True
#
# POSE_HIGHER_RESOLUTION_NET.STAGE1 = CN()
# POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_MODULES = 1
# POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BRANCHES = 1
# POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BLOCKS = [4]
# POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_CHANNELS = [64]
# POSE_HIGHER_RESOLUTION_NET.STAGE1.BLOCK = 'BOTTLENECK'
# POSE_HIGHER_RESOLUTION_NET.STAGE1.FUSE_METHOD = 'SUM'
#
# POSE_HIGHER_RESOLUTION_NET.STAGE2 = CN()
# POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
# POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
# POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
# POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [48, 96]
# POSE_HIGHER_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
# POSE_HIGHER_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'
#
# POSE_HIGHER_RESOLUTION_NET.STAGE3 = CN()
# POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_MODULES = 4
# POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
# POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
# POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [48, 96, 192]
# POSE_HIGHER_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
# POSE_HIGHER_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'
#
# POSE_HIGHER_RESOLUTION_NET.STAGE4 = CN()
# POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_MODULES = 3
# POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
# POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
# POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
# POSE_HIGHER_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
# POSE_HIGHER_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'
#
# POSE_HIGHER_RESOLUTION_NET.LOSS = CN()
# POSE_HIGHER_RESOLUTION_NET.LOSS.WITH_AE_LOSS = [False, False, False]
#
# POSE_HIGHER_RESOLUTION_NET.OCR = CN()
# POSE_HIGHER_RESOLUTION_NET.OCR.DROPOUT = 0.05
# POSE_HIGHER_RESOLUTION_NET.OCR.KEY_CHANNELS = 48
# POSE_HIGHER_RESOLUTION_NET.OCR.MID_CHANNELS = 96
# POSE_HIGHER_RESOLUTION_NET.OCR.SCALE = 1

# small net
POSE_HIGHER_RESOLUTION_NET = CN()
POSE_HIGHER_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGHER_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGHER_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
POSE_HIGHER_RESOLUTION_NET.NUM_JOINTS = 2
POSE_HIGHER_RESOLUTION_NET.TAG_PER_JOINT = True

POSE_HIGHER_RESOLUTION_NET.STAGE1 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BRANCHES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BLOCKS = [2]
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_CHANNELS = [64]
POSE_HIGHER_RESOLUTION_NET.STAGE1.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE1.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [2, 2]
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
POSE_HIGHER_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
POSE_HIGHER_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_MODULES = 4
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [2, 2, 2]
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
POSE_HIGHER_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
POSE_HIGHER_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_MODULES = 3
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
POSE_HIGHER_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
POSE_HIGHER_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.LOSS = CN()
POSE_HIGHER_RESOLUTION_NET.LOSS.WITH_AE_LOSS = [False, False, False]

POSE_HIGHER_RESOLUTION_NET.OCR = CN()
POSE_HIGHER_RESOLUTION_NET.OCR.DROPOUT = 0.05
POSE_HIGHER_RESOLUTION_NET.OCR.KEY_CHANNELS = 48
POSE_HIGHER_RESOLUTION_NET.OCR.MID_CHANNELS = 96
POSE_HIGHER_RESOLUTION_NET.OCR.SCALE = 1

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def crop_or_pad2d(tensor, target_size):
    _, _, tensor_h, tensor_w = tensor.size()
    diff_h = (tensor_h - target_size[2])
    diff_w = (tensor_w - target_size[3])

    # Crop
    if diff_h > 0 or diff_w > 0:
        # from_h, from_w = diff_h // 2, diff_w // 2
        # to_h = target_size[0] + from_h
        # to_w = target_size[1] + from_w

        top = torch.floor(diff_h.to(torch.float32) / 2)
        bottom = torch.ceil(diff_h.to(torch.float32) / 2) + target_size
        left = torch.floor(diff_w.to(torch.float32) / 2)
        right = torch.ceil(diff_w.to(torch.float32) / 2) + target_size
        tensor = tensor[:, :, top: bottom, left: right]

    if diff_h < 0 or diff_w < 0:
        nn.functional.pad(tensor, )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample if downsample is not None else torch.nn.Identity()
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def fuse(self):
        torch.quantization.fuse_modules(self,
                                        [
                                            ['conv1', 'bn1', 'relu1'],
                                            ['conv2', 'bn2']
                                        ],
                                        inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.skip_add.add_relu(out, residual)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes: num channels in input tensor
        # planes: num channels in bottleneck
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.downsample = downsample if downsample is not None else torch.nn.Identity()
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def fuse(self):
        torch.quantization.fuse_modules(self,
                                        [
                                            ['conv1', 'bn1', 'relu1'],
                                            ['conv2', 'bn2', 'relu2'],
                                            ['conv3', 'bn3']
                                        ],
                                        inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.skip_add.add_relu(out, residual)
        return out


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def fuse(self):
        torch.quantization.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def fuse(self):
        torch.quantization.fuse_modules(self, [['conv', 'bn']], inplace=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return x


class ConvTransposeBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvTransposeBN, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    # def fuse(self):
    #     torch.quantization.fuse_modules(self, [['conv_transpose', 'bn']], inplace=True)

    def forward(self, input):
        x = self.conv_transpose(input)
        x = self.bn(x)
        return x


class HighResolutionModule(nn.Module):
    # This module builds num_branches of conv layers for each resolution and
    # builds fuse layers to allow branches exchange information
    def __init__(self,
                 num_in_channels,
                 num_out_channels,
                 num_blocks,
                 block,
                 fuse_method):
        super(HighResolutionModule, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.fuse_method = fuse_method
        self.num_branches = len(num_out_channels)

        self.fuse_module = TransitionFuse(num_in_channels, num_out_channels)
        # self.fuse_module = ImprovedTransitionFuseV4(num_in_channels, num_out_channels)
        self.branches: Final = self._make_branches(self.num_branches, block, num_blocks, num_out_channels)

    def _make_one_branch(self, block, num_blocks, num_channels):
        layers = []
        num_in_channels = num_channels * block.expansion
        for i in range(0, num_blocks):
            # layers.append(block(num_in_channels, num_channels))
            layers.append(block(num_in_channels, num_channels))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        # This funcrion builds list of parallel sequences of convolution blocks where each item in this list is
        # responsible for processing specific single scale
        branches = []

        for branch_index in range(num_branches):
            branches.append(self._make_one_branch(block, num_blocks[branch_index],
                                                  num_channels[branch_index] // block.expansion))

        return nn.ModuleList(branches)

    def forward(self, x: List[torch.Tensor]):
        x = self.fuse_module(x)

        branch_idx = 0
        for branch_module in self.branches:
            x[branch_idx] = branch_module(x[branch_idx])
            branch_idx += 1
        return x


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': BottleneckBlock
}


class Stage(nn.Module):
    __constants__ = ['mods']

    def __init__(self, num_in_channels, num_out_channels, num_modules, num_blocks, block, fuse_method):
        super(Stage, self).__init__()

        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_in_channels,
                    num_out_channels,
                    num_blocks,
                    block,
                    fuse_method)
            )
            num_in_channels = num_out_channels
        self.num_out_channels = [num_channels * block.expansion for num_channels in num_out_channels]
        self.mods = nn.ModuleList(modules)

    def forward(self, input: List[torch.Tensor]):
        x = input
        for module in self.mods:
            x = module(x)
        return x


class TransitionFuse(nn.Module):
    # This module can grow new branch and allows branches to exchange data
    # It should be placed before stage of HRBlocks and after last stage in HRNet
    __constants__ = ['fuse_layers']

    def __init__(self, num_in_channels, num_out_channels):
        # num_in_channels is a list of input depth for each resolution
        # num_out_channels is a list of output depth for each resolution
        super(TransitionFuse, self).__init__()

        self.num_in_branches = len(num_in_channels)
        self.num_out_branches = len(num_out_channels)

        fuse_layers = []
        # `i` is output branch idx
        # `j` is input branch idx
        # branch 0 is largest resolution
        for i in range(len(num_out_channels)):
            fuse_layer = []
            for j in range(len(num_in_channels)):
                if j > i:
                    # Upsample smaller scale
                    fuse_layer.append(
                        ConvBN(num_in_channels[j],
                               num_out_channels[i],
                               kernel_size=1,
                               stride=1,
                               padding=0),
                        # nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                    )
                elif j == i:
                    # Same resolution
                    # Add conv+bn if num of channels changed
                    if num_in_channels[j] == num_out_channels[i]:
                        fuse_layer.append(nn.Identity())
                    else:
                        fuse_layer.append(
                            ConvBN(num_in_channels[j],
                                   num_out_channels[i],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0))
                else:
                    conv3x3s = []
                    for k in range(j, i):
                        if k == i - 1:
                            # Last layer of subsampling branch
                            conv3x3s.append(
                                ConvBN(num_in_channels[k],
                                       num_out_channels[k + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1))
                        else:
                            # Intermediate layer of subsampling branch
                            conv3x3s.append(
                                ConvBNRelu(num_in_channels[k],
                                           num_in_channels[k + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        self.fuse_layers = nn.ModuleList(fuse_layers)

        self.adds = []
        for j in range(len(num_out_channels)):
            additions_for_scale = []
            for i in range(len(num_in_channels) - 1):
                additions_for_scale.append(nn.quantized.FloatFunctional())
            self.adds.append(nn.ModuleList(additions_for_scale))
        self.adds = nn.ModuleList(self.adds)

    def forward(self, input: List[torch.Tensor]):
        # input: list of tensors of different resolution

        x_fuse: List[List[torch.Tensor]] = []
        output_idx = 0
        for scale_fuse_layers in self.fuse_layers:
            tensors_to_fuse = []
            input_idx = 0
            for fuse_layer in scale_fuse_layers:
                resized_tensor = fuse_layer(input[input_idx])
                if output_idx <= input_idx and resized_tensor.size() != input[output_idx].size():
                    resized_tensor = torch.nn.functional.interpolate(resized_tensor,
                                                                     size=input[output_idx].size()[2:4],
                                                                     mode='bilinear',
                                                                     align_corners=False)
                tensors_to_fuse.append(resized_tensor)
                input_idx += 1
            x_fuse.append(tensors_to_fuse)
            output_idx += 1

            # y = torch.sum(torch.stack(tensors_to_fuse, dim=0), dim=0)
            # y = tensors_to_fuse[0]
            # tensor_idx = 0
            # for add in self.adds:
            #     y = add.add(y, tensors_to_fuse[tensor_idx + 1])
            #     tensor_idx += 1
            #

            # x_fuse.append(torch.nn.functional.relu(y, inplace=True))

        output = []
        output_idx = 0
        for additions_for_scale in self.adds:
            y = x_fuse[output_idx][0]
            input_idx = 1
            for addition in additions_for_scale:
                if input_idx != self.num_in_branches - 1:
                    y = addition.add(y, x_fuse[output_idx][input_idx])
                else:
                    y = addition.add_relu(y, x_fuse[output_idx][input_idx])
                input_idx += 1
            output_idx += 1
            output.append(y)
        return output


class ImprovedTransitionFuse(nn.Module):
    # This module can grow new branch and allows branches to exchange data
    # It should be placed before stage of HRBlocks and after last stage in HRNet

    def __init__(self, num_in_channels, num_out_channels):
        # num_in_channels is a list of input depth for each resolution
        # num_out_channels is a list of output depth for each resolution
        super(ImprovedTransitionFuse, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_in_branches = len(num_in_channels)
        self.num_out_branches = len(num_out_channels)
        self.num_stages = max(self.num_in_branches, self.num_out_branches)

        self.upsampling_layers = []
        for i in range(len(num_in_channels) - 1):
            in_channels = num_in_channels[i + 1] if i == len(num_in_channels) - 2 or i + 1 >= len(num_out_channels) else \
                num_out_channels[i + 1]
            out_channels = num_out_channels[i] if i < len(num_out_channels) else num_in_channels[i]
            self.upsampling_layers.append(
                nn.Sequential(
                    nn.ReLU(inplace=False) if i > 0 else nn.Identity(),
                    ConvTransposeBN(in_channels,
                                    out_channels,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)))
        self.upsampling_layers = nn.ModuleList(reversed(self.upsampling_layers))

        self.downsampling_layers = []
        for i in range(len(num_out_channels) - 1):
            in_channels = num_in_channels[i] if i == 0 else num_out_channels[i]
            self.downsampling_layers.append(
                nn.Sequential(
                    nn.ReLU(inplace=False) if i > 0 else nn.Identity(),
                    ConvBN(
                        in_channels,
                        num_out_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1)))
        self.downsampling_layers = nn.ModuleList(self.downsampling_layers)

        self.identity_layers = []
        for i in range(min(len(num_in_channels), len(num_out_channels))):
            if num_in_channels[i] != num_out_channels[i]:
                self.identity_layers.append(
                    nn.Sequential(
                        ConvBN(num_in_channels[i],
                               num_out_channels[i],
                               kernel_size=1,
                               stride=1,
                               padding=0)))
            else:
                self.identity_layers.append(nn.Identity())
        self.identity_layers = nn.ModuleList(self.identity_layers)

    def forward(self, input: List[torch.Tensor]):
        # input: list of tensors of different resolution

        for stage_idx in range(1, self.num_stages):
            downsample_idx = 1
            for downsample_module in self.downsampling_layers:
                if downsample_idx == stage_idx:
                    if downsample_idx < self.num_in_branches:
                        downsampled = downsample_module(input[downsample_idx - 1])

                        identity_idx = 0
                        for identity_module in self.identity_layers:
                            if identity_idx == downsample_idx and downsampled.size(1) != input[downsample_idx].size(1):
                                input[identity_idx] = identity_module(input[identity_idx])
                            identity_idx += 1

                        input[downsample_idx] = input[downsample_idx] + downsampled
                    else:
                        input.append(downsample_module(input[downsample_idx - 1]))
                downsample_idx += 1

            upsample_idx = self.num_in_branches - 2
            for upsample_module in self.upsampling_layers:
                if upsample_idx == self.num_in_branches - stage_idx:
                    upsampled = upsample_module(input[upsample_idx + 1])

                    identity_idx = 0
                    for identity_module in self.identity_layers:
                        if identity_idx == upsample_idx and upsampled.size(1) != input[upsample_idx].size(1):
                            input[identity_idx] = identity_module(input[identity_idx])
                        identity_idx += 1

                    if input[upsample_idx].shape[2:4] != upsampled.shape[2:4]:
                        upsampled = torch.nn.functional.interpolate(upsampled, input[upsample_idx].shape[2:4],
                                                                    mode='bilinear',
                                                                    align_corners=True)
                    input[upsample_idx] = input[upsample_idx] + upsampled

                upsample_idx -= 1

        identity_idx = 0
        for identity_module in self.identity_layers:
            if input[identity_idx].size(1) != self.num_out_channels[identity_idx]:
                input[identity_idx] = identity_module(input[identity_idx])
            identity_idx += 1

        for i in range(len(input)):
            input[i] = torch.nn.functional.relu(input[i])

        return input


class DownsampleModule(nn.Module):
    def __init__(self, high_res_in_channels, in_channels, out_channels, use_activation=True):
        super(DownsampleModule, self).__init__()
        self.preactivation_relu = nn.ReLU(inplace=False) if use_activation else nn.Identity()
        self.downsample_conv_bn = ConvBN(high_res_in_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
        self.identity_conv_bn = ConvBN(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0) if in_channels != out_channels else nn.Identity()
        self.add = nn.quantized.FloatFunctional()

    def forward(self, high_res, low_res):
        x = self.preactivation_relu(high_res)
        x = self.downsample_conv_bn(x)
        identity = self.identity_conv_bn(low_res)
        x = self.add.add(x, identity)
        return x


class UpsampleModule(nn.Module):
    def __init__(self, low_res_in_channels, in_channels, out_channels, use_activation=True):
        super(UpsampleModule, self).__init__()
        self.preactivation_relu = nn.ReLU(inplace=False) if use_activation else nn.Identity()
        self.upsample_conv_bn = ConvBN(low_res_in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.identity_conv_bn = ConvBN(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0) if in_channels != out_channels else nn.Identity()
        self.add = nn.quantized.FloatFunctional()

    def forward(self, low_res, high_res):
        x = self.preactivation_relu(low_res)
        x = torch.nn.functional.interpolate(x, size=high_res.shape[2:4], mode='bilinear', align_corners=False)
        x = self.upsample_conv_bn(x)
        identity = self.identity_conv_bn(high_res)
        x = self.add.add(x, identity)
        return x


# class DownsampleConcatModule(nn.Module):
#     def __init__(self, high_res_in_channels, in_channels, out_channels, use_activation):
#         super(DownsampleConcatModule, self).__init__()
#         self.high_res_relu = nn.ReLU(inplace=False) if use_activation else nn.Identity()
#         self.low_res_relu = nn.ReLU(inplace=False)
#         self.downsample_conv_bn = ConvBN(high_res_in_channels,
#                                          out_channels,
#                                          kernel_size=3,
#                                          stride=2,
#                                          padding=1)
#         self.identity_conv_bn = ConvBN(in_channels + out_channels,
#                                        out_channels,
#                                        kernel_size=1,
#                                        stride=1,
#                                        padding=0)
#         self.concat = nn.quantized.FloatFunctional()
#
#     def forward(self, high_res, low_res):
#         x = self.high_res_relu(high_res)
#         low_res = self.low_res_relu(low_res)
#         x = self.downsample_conv_bn(x)
#         x = self.concat.cat([x, low_res], dim=1)
#         x = self.identity_conv_bn(x)
#         return x
#
#
# class UpsampleConcatModule(nn.Module):
#     def __init__(self, low_res_in_channels, in_channels, out_channels, use_activation):
#         super(UpsampleConcatModule, self).__init__()
#         self.low_res_relu = nn.ReLU(inplace=False) if use_activation else nn.Identity()
#         self.high_res_relu = nn.ReLU(inplace=False)
#         self.upsample_conv_bn = ConvBN(low_res_in_channels + in_channels,
#                                        out_channels,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#         self.concat = nn.quantized.FloatFunctional()
#
#     def forward(self, low_res, high_res):
#         low_res = self.low_res_relu(low_res)
#         high_res = self.high_res_relu(high_res)
#         x = torch.nn.functional.interpolate(low_res, size=high_res.shape[2:4], mode='bilinear', align_corners=False)
#         x = self.concat.cat([x, high_res], dim=1)
#         x = self.upsample_conv_bn(x)
#         return x


class ImprovedTransitionFuseV2(nn.Module):
    # This module can grow new branch and allows branches to exchange data
    # It should be placed before stage of HRBlocks and after last stage in HRNet

    def __init__(self, num_in_channels, num_out_channels):
        # num_in_channels is a list of input depth for each resolution
        # num_out_channels is a list of output depth for each resolution
        super(ImprovedTransitionFuseV2, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_in_branches = len(num_in_channels)
        self.num_out_branches = len(num_out_channels)
        self.num_stages = max(self.num_in_branches, self.num_out_branches)

        for i in range(self.num_in_branches):
            if i > self.num_out_branches - 1:
                self.num_out_channels.append(num_in_channels[i])

        self.upsampling_layers = []
        for i in range(self.num_in_branches - 1):
            self.upsampling_layers.append(
                UpsampleModule(
                    low_res_in_channels=num_in_channels[i + 1],
                    in_channels=num_in_channels[i],
                    out_channels=num_out_channels[i],
                    use_activation=False if i == 0 else True))
        self.upsampling_layers = nn.ModuleList(reversed(self.upsampling_layers))

        self.downsampling_layers = []
        for i in range(self.num_out_branches - 1):
            if i < len(num_in_channels) - 1:
                self.downsampling_layers.append(
                    DownsampleModule(
                        high_res_in_channels=num_in_channels[i],
                        in_channels=num_in_channels[i + 1],
                        out_channels=num_out_channels[i + 1],
                        use_activation=False if i == 0 else True))
            else:
                self.downsampling_layers.append(
                    nn.Sequential(
                        nn.ReLU(inplace=False),
                        ConvBN(in_channels=num_out_channels[i],
                               out_channels=num_out_channels[i + 1],
                               kernel_size=3,
                               stride=2,
                               padding=1)
                    )
                )
        self.downsampling_layers = nn.ModuleList(self.downsampling_layers)

        self.branch_grow_layers = []

        self.identity_highest = ConvBN(
            num_in_channels[0],
            num_out_channels[0],
            kernel_size=1,
            stride=1,
            padding=0)  # if num_in_channels[0] != num_out_channels[0] else nn.Identity()

        self.identity_lowest = ConvBN(
            num_in_channels[-1],
            num_out_channels[-1],
            kernel_size=1,
            stride=1,
            padding=0) if len(num_in_channels) == len(num_out_channels) else None

    def forward(self, input: List[torch.Tensor]):
        # input: list of tensors of different resolution

        downsampled_tensors = []
        downsampled_tensors.append(self.identity_highest(input[0]))
        downsample_idx = 1
        for downsample_module in self.downsampling_layers:
            if downsample_idx < self.num_in_branches:
                downsampled_tensors.append(downsample_module(input[downsample_idx - 1], input[downsample_idx]))
            else:
                downsampled_tensors.append(downsample_module(downsampled_tensors[downsample_idx - 1]))
            downsample_idx += 1

        upsampled_tensors = []
        if self.num_in_branches == self.num_out_branches:
            upsampled_tensors.append(self.identity_lowest(input[-1]))

        upsample_idx = self.num_in_branches - 2
        for upsample_module in self.upsampling_layers:
            if upsample_idx == self.num_in_branches - 2:
                upsampled_tensors.insert(0, upsample_module(input[upsample_idx + 1], input[upsample_idx]))
            else:
                upsampled_tensors.insert(0, upsample_module(upsampled_tensors[0], input[upsample_idx]))
            upsample_idx -= 1

        output = []
        for i in range(self.num_out_branches):
            if i < len(upsampled_tensors):
                output.append(downsampled_tensors[i] + upsampled_tensors[i])
            else:
                output.append(downsampled_tensors[i])

        for i in range(len(output)):
            output[i] = torch.nn.functional.relu(output[i])

        return output


class ImprovedTransitionFuseV3(nn.Module):
    # This module can grow new branch and allows branches to exchange data
    # It should be placed before stage of HRBlocks and after last stage in HRNet

    def __init__(self, num_in_channels, num_out_channels):
        # num_in_channels is a list of input depth for each resolution
        # num_out_channels is a list of output depth for each resolution
        super(ImprovedTransitionFuseV3, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_in_branches = len(num_in_channels)
        self.num_out_branches = len(num_out_channels)
        self.num_stages = max(self.num_in_branches, self.num_out_branches)

        for i in range(self.num_in_branches):
            if i > self.num_out_branches - 1:
                self.num_out_channels.append(num_in_channels[i])

        self.upsampling_layers = []
        for i in range(self.num_in_branches - 1):
            self.upsampling_layers.append(
                UpsampleModule(
                    low_res_in_channels=num_in_channels[i + 1],
                    in_channels=num_in_channels[i],
                    out_channels=num_out_channels[i],
                    use_activation=False if i == 0 else True))
        self.upsampling_layers = nn.ModuleList(reversed(self.upsampling_layers))

        self.downsampling_layers = []
        for i in range(self.num_in_branches - 1):
            self.downsampling_layers.append(
                DownsampleModule(
                    high_res_in_channels=num_in_channels[i],
                    in_channels=num_in_channels[i + 1],
                    out_channels=num_out_channels[i + 1],
                    use_activation=False if i == 0 else True))

        self.downsampling_layers = nn.ModuleList(self.downsampling_layers)

        self.branch_grow_layers = []
        for i in range(self.num_in_branches - 1, max(self.num_in_branches, self.num_out_branches) - 1):
            self.branch_grow_layers.append(
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    ConvBN(in_channels=num_out_channels[i],
                           out_channels=num_out_channels[i + 1],
                           kernel_size=3,
                           stride=2,
                           padding=1)
                )
            )
        self.branch_grow_layers = nn.ModuleList(self.branch_grow_layers)

        self.identity_highest = ConvBN(
            num_in_channels[0],
            num_out_channels[0],
            kernel_size=1,
            stride=1,
            padding=0)  # if num_in_channels[0] != num_out_channels[0] else nn.Identity()

        self.identity_lowest = ConvBN(
            num_in_channels[-1],
            num_out_channels[-1],
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, input: List[torch.Tensor]):
        # input: list of tensors of different resolution

        downsampled_tensors = []
        downsampled_tensors.append(self.identity_highest(input[0]))
        downsample_idx = 1
        for downsample_module in self.downsampling_layers:
            downsampled_tensors.append(downsample_module(input[downsample_idx - 1], input[downsample_idx]))
            downsample_idx += 1

        for downsample_module in self.branch_grow_layers:
            downsampled_tensors.append(downsample_module(downsampled_tensors[-1]))

        upsampled_tensors = []
        if self.num_in_branches == self.num_out_branches:
            upsampled_tensors.append(self.identity_lowest(input[-1]))

        upsample_idx = self.num_in_branches - 2
        for upsample_module in self.upsampling_layers:
            if upsample_idx == self.num_in_branches - 2:
                upsampled_tensors.insert(0, upsample_module(input[upsample_idx + 1], input[upsample_idx]))
            else:
                upsampled_tensors.insert(0, upsample_module(upsampled_tensors[0], input[upsample_idx]))
            upsample_idx -= 1

        output = []
        for i in range(self.num_out_branches):
            if i < len(upsampled_tensors):
                output.append(downsampled_tensors[i] + upsampled_tensors[i])
            else:
                output.append(downsampled_tensors[i])

        for i in range(len(output)):
            output[i] = torch.nn.functional.relu(output[i])

        return output


class ImprovedTransitionFuseV4(nn.Module):
    # This module can grow new branch and allows branches to exchange data
    # It should be placed before stage of HRBlocks and after last stage in HRNet

    def __init__(self, num_in_channels, num_out_channels):
        # num_in_channels is a list of input depth for each resolution
        # num_out_channels is a list of output depth for each resolution
        super(ImprovedTransitionFuseV4, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = []
        self.num_in_branches = len(num_in_channels)
        self.num_out_branches = len(num_out_channels)

        for i in range(max(self.num_in_branches, self.num_out_branches)):
            if i > self.num_out_branches - 1:
                self.num_out_channels.append(num_in_channels[i])
            else:
                self.num_out_channels.append(num_out_channels[i])

        self.upsampling_layers = []
        for i in range(self.num_in_branches - 1):
            self.upsampling_layers.append(
                UpsampleModule(
                    low_res_in_channels=self.num_out_channels[i + 1] if i < self.num_in_branches - 2 else
                    self.num_in_channels[i + 1],
                    in_channels=self.num_in_channels[i],
                    out_channels=self.num_out_channels[i]))
        self.upsampling_layers = nn.ModuleList(reversed(self.upsampling_layers))

        self.downsampling_layers = []
        for i in range(min(self.num_in_branches - 1, self.num_out_branches - 1)):
            self.downsampling_layers.append(
                DownsampleModule(
                    high_res_in_channels=self.num_in_channels[i] if i == 0 else self.num_out_channels[i],
                    in_channels=self.num_in_channels[i + 1],
                    out_channels=self.num_out_channels[i + 1]))

        self.downsampling_layers = nn.ModuleList(self.downsampling_layers)

        self.branch_grow_layers = []
        for i in range(self.num_in_branches - 1, max(self.num_in_branches, self.num_out_branches) - 1):
            self.branch_grow_layers.append(
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    ConvBN(in_channels=self.num_in_channels[i] if len(self.downsampling_layers) == 0 else
                    self.num_out_channels[i],
                           out_channels=self.num_out_channels[i + 1],
                           kernel_size=3,
                           stride=2,
                           padding=1)
                )
            )
        self.branch_grow_layers = nn.ModuleList(self.branch_grow_layers)

        self.identity_highest = ConvBN(
            self.num_in_channels[0],
            self.num_out_channels[0],
            kernel_size=1,
            stride=1,
            padding=0) if self.num_in_channels[0] != self.num_out_channels[0] and self.num_in_branches == 1 else nn.Identity()

        self.add_modules = []
        for i in range(self.num_out_branches):
            self.add_modules.append(torch.nn.quantized.FloatFunctional())

    def forward(self, input: List[torch.Tensor]):
        # input: list of tensors of different resolution

        downsampled_tensors = []
        downsample_idx = 1
        for downsample_module in self.downsampling_layers:
            if downsample_idx == 1:
                downsampled_tensors.append(downsample_module(input[downsample_idx - 1], input[downsample_idx]))
            else:
                downsampled_tensors.append(downsample_module(downsampled_tensors[-1], input[downsample_idx]))
            downsample_idx += 1

        for downsample_module in self.branch_grow_layers:
            if len(downsampled_tensors) == 0:
                downsampled_tensors.append(downsample_module(input[0]))
            else:
                downsampled_tensors.append(downsample_module(downsampled_tensors[-1]))

        upsampled_tensors = []
        upsample_idx = self.num_in_branches - 2
        for upsample_module in self.upsampling_layers:
            if upsample_idx == self.num_in_branches - 2:
                upsampled_tensors.insert(0, upsample_module(input[upsample_idx + 1], input[upsample_idx]))
            else:
                upsampled_tensors.insert(0, upsample_module(upsampled_tensors[0], input[upsample_idx]))
            upsample_idx -= 1

        output = []

        for i in range(self.num_out_branches):
            # out_idx = 0
            # for module in self.add_modules:
            if i == 0 and len(upsampled_tensors) == 0:
                output.append(self.identity_highest(input[i]))
            elif i == 0 and len(upsampled_tensors) > 0:
                output.append(upsampled_tensors[i])
            elif i < len(upsampled_tensors):
                output.append(downsampled_tensors[i - 1] + upsampled_tensors[i])
            else:
                output.append(downsampled_tensors[i - 1])

        # TODO: In init build module list of add ops
        for i in range(len(output)):
            output[i] = torch.nn.functional.relu(output[i])

        return output


class HighResolutionMultiscaleAggregator(nn.Module):
    def __init__(self):
        super(HighResolutionMultiscaleAggregator, self).__init__()
        self.concat = torch.nn.quantized.FloatFunctional()

    def forward(self, input: List[torch.Tensor]):
        tgt_size = input[0].size()[2:4]

        resized_list = []
        for scale_tensor in input:
            resized_list.append(torch.nn.functional.interpolate(scale_tensor, tgt_size, mode='bilinear',
                                                                align_corners=False))

        output = self.concat.cat(resized_list, dim=1)
        return output

class HighResolutionMultiscaleAggregatorWithPixelwiseAttention(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(HighResolutionMultiscaleAggregator, self).__init__()
        self.concat = torch.nn.quantized.FloatFunctional()
        self.attention_modules = nn.ModuleList()
        for i in range(len(input_channels) - 1):
            self.attention_modules = nn.Sequential(
                ConvBNRelu(input_channels[i], input_channels[i], kernel_size=3, stride=1, padding=1),
                ConvBNRelu(input_channels[i], input_channels[i], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(input_channels[i], num_classes, kernel_size=1, stride=1, padding=0)
            )
        self.float_adds = nn.ModuleList()
        for i in range(len(input_channels) - 1):
            self.float_adds.append(nn.quantized.FloatFunctional())

        self.float_multiplies = nn.ModuleList()
        for i in len(len(input_channels) - 1):
            self.float_multiplies.append(nn.ModuleList([
                nn.quantized.FloatFunctional(),
                nn.quantized.FloatFunctional()
            ]))

    def forward(self, input: List[torch.Tensor]):
        attention_masks = []
        i = len(input) - 1
        for attention_module in self.attention_modules:
            attention_masks.append(attention_module(input[i]))
            i -= 1

        # for
        tgt_size = input[0].size()[2:4]

        resized_list = []
        for scale_tensor in input:
            resized_list.append(torch.nn.functional.interpolate(scale_tensor, tgt_size, mode='bilinear',
                                                                align_corners=False))

        output = self.concat.cat(resized_list, dim=1)
        return output


class HigherDecoderStage(nn.Module):
    # layers: Final
    # cat_output: Final

    def __init__(self, input_channels, output_channels, final_kernel_size, do_deconv, deconv_num_basic_blocks,
                 deconv_output_channels, deconv_kernel_size, cat_output):
        super(HigherDecoderStage, self).__init__()
        layers = []
        self.cat_output = cat_output
        self.concat = torch.nn.quantized.FloatFunctional()

        if do_deconv:
            deconv_input_channels = input_channels
            deconv_kernel_size, deconv_padding, deconv_output_padding = self._get_deconv_cfg(deconv_kernel_size)
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=deconv_input_channels,
                        out_channels=deconv_output_channels,
                        kernel_size=deconv_kernel_size,
                        stride=2,
                        padding=deconv_padding,
                        output_padding=deconv_output_padding,
                        bias=False),
                    nn.BatchNorm2d(deconv_output_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))

            for i in range(deconv_num_basic_blocks):
                layers.append(nn.Sequential(
                    BasicBlock(deconv_output_channels, deconv_output_channels),
                ))

            classifier_input_channels = deconv_output_channels
        else:
            layers.append(nn.Identity())
            classifier_input_channels = input_channels

        self.layers = nn.Sequential(*layers)

        self.output_layer = nn.Conv2d(in_channels=classifier_input_channels,
                                      out_channels=output_channels,
                                      kernel_size=final_kernel_size,
                                      stride=1,
                                      padding=1 if final_kernel_size == 3 else 0)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def forward(self, input: torch.Tensor):
        features = self.layers(input)
        output = self.output_layer(features)
        if self.cat_output:
            features = self.concat.cat([features, output], dim=1)

        return output, features


class HigherDecoder(nn.Module):
    # decoder_layers: Final

    def __init__(self, input_channels, output_channels, final_kernel_size, num_deconvs, deconv_num_basic_blocks,
                 deconv_output_channels, deconv_kernel_size, cat_output):
        super(HigherDecoder, self).__init__()
        decoder_layers = []
        decoder_layers.append(
            HigherDecoderStage(input_channels=input_channels,
                               output_channels=output_channels,
                               final_kernel_size=final_kernel_size,
                               do_deconv=False,
                               deconv_num_basic_blocks=None,
                               deconv_output_channels=None,
                               deconv_kernel_size=None,
                               cat_output=cat_output[0])
        )

        prev_feature_channels = input_channels + output_channels if cat_output else input_channels
        for i in range(num_deconvs):
            decoder_layers.append(
                HigherDecoderStage(input_channels=prev_feature_channels,
                                   output_channels=output_channels,
                                   final_kernel_size=final_kernel_size,
                                   do_deconv=True,
                                   deconv_num_basic_blocks=deconv_num_basic_blocks,
                                   deconv_output_channels=deconv_output_channels[i],
                                   deconv_kernel_size=deconv_kernel_size[i],
                                   cat_output=cat_output[i])
            )
            prev_feature_channels = deconv_output_channels[i] + output_channels \
                if cat_output else deconv_output_channels

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, input: torch.Tensor):
        output: List[torch.Tensor] = []
        features = input
        for decoder_stage in self.decoder_layers:
            stage_output, features = decoder_stage(features)
            output.append(stage_output)
        return output


class Stem(nn.Module):
    def __init__(self, num_in_channels=3, num_out_channels=64):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_out_channels, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_out_channels, num_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_out_channels, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=True)

    def fuse(self):
        torch.quantization.fuse_modules(self,
                                        [
                                            ['conv1', 'bn1', 'relu1'],
                                            ['conv2', 'bn2', 'relu2'],
                                        ],
                                        inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class HigherResolutionNet(nn.Module):
    __constants__ = ['stage2', 'stage3', 'stage4']

    def __init__(self, cfg):
        super(HigherResolutionNet, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # self.dequant_features = torch.quantization.DeQuantStub()

        # stem net
        stem_num_channels = cfg['STEM_INPLANES']
        self.stem = Stem(num_in_channels=3, num_out_channels=stem_num_channels)

        self.stage1_cfg = cfg['STAGE1']
        num_modules = self.stage1_cfg['NUM_MODULES']
        num_blocks = self.stage1_cfg['NUM_BLOCKS']
        num_channels = self.stage1_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        fuse_method = self.stage1_cfg['FUSE_METHOD']
        num_channels = [channels * block.expansion for channels in num_channels]
        self.stage1 = Stage(num_in_channels=[stem_num_channels],
                            num_out_channels=num_channels,
                            num_modules=num_modules,
                            num_blocks=num_blocks,
                            block=block,
                            fuse_method=fuse_method)

        self.stage2_cfg = cfg['STAGE2']
        num_in_channels = [channels * block.expansion for channels in num_channels]
        num_modules = self.stage2_cfg['NUM_MODULES']
        num_blocks = self.stage2_cfg['NUM_BLOCKS']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        fuse_method = self.stage2_cfg['FUSE_METHOD']
        num_channels = [channels * block.expansion for channels in num_channels]
        # TODO: Replace hardcoded 256 with calculation of number of channels based on config
        self.stage2 = Stage(num_in_channels=[64 * 4],
                            num_out_channels=num_channels,
                            num_modules=num_modules,
                            num_blocks=num_blocks,
                            block=block,
                            fuse_method=fuse_method)
        pre_stage_channels = num_channels

        self.stage3_cfg = cfg['STAGE3']
        num_modules = self.stage3_cfg['NUM_MODULES']
        num_blocks = self.stage3_cfg['NUM_BLOCKS']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        fuse_method = self.stage3_cfg['FUSE_METHOD']
        num_channels = [channels * block.expansion for channels in num_channels]
        self.stage3 = Stage(num_in_channels=pre_stage_channels,
                            num_out_channels=num_channels,
                            num_modules=num_modules,
                            num_blocks=num_blocks,
                            block=block,
                            fuse_method=fuse_method)
        pre_stage_channels = num_channels

        self.stage4_cfg = cfg['STAGE4']
        num_modules = self.stage4_cfg['NUM_MODULES']
        num_blocks = self.stage4_cfg['NUM_BLOCKS']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        fuse_method = self.stage4_cfg['FUSE_METHOD']
        num_channels = [channels * block.expansion for channels in num_channels]
        self.stage4 = Stage(num_in_channels=pre_stage_channels,
                            num_out_channels=num_channels,
                            num_modules=num_modules,
                            num_blocks=num_blocks,
                            block=block,
                            fuse_method=fuse_method)
        pre_stage_channels = num_channels

        # num_final_channels = [pre_stage_channels[0]]
        num_final_channels = sum(pre_stage_channels)


        # self.final_aggregation = TransitionFuse(pre_stage_channels, pre_stage_channels)
        self.multires_aggregation = HighResolutionMultiscaleAggregator()
        self.cls_head = nn.Sequential(
                ConvBNRelu(num_final_channels, num_final_channels // 2, kernel_size=3, stride=1, padding=1),
                ConvBNRelu(num_final_channels // 2, num_final_channels // 4, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(num_final_channels // 4, cfg.NUM_JOINTS, kernel_size=1, stride=1, padding=0, bias=True)
            )


        # self.feature_dequants = nn.ModuleList()
        # for _ in self.stage4.num_out_channels:
        #     self.feature_dequants.append(torch.quantization.DeQuantStub())

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)

        y_list: List[torch.Tensor] = [x]

        y_list = self.stage1(y_list)
        y_list = self.stage2(y_list)
        y_list = self.stage3(y_list)
        y_list = self.stage4(y_list)

        # features = y_list = self.final_aggregation(y_list)

        features = multires_features = self.multires_aggregation(y_list)

        logits = self.cls_head(multires_features)

        outputs = [self.dequant(logits)]
        # outputs = [out]
        # features = [self.dequant_features(features)]
        features = [features]
        return features, outputs

    def fuse_model(self):
        def fuse_fn(m):
            if hasattr(m, 'fuse') and callable(m.fuse):
                m.fuse()

        self.apply(fuse_fn)

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)

    def init_module(self, module):
        logger.info('=> init weights from normal distribution')
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def reinit_classifier(self):
        self.init_module(self.aux_head)
        self.init_module(self.ocr_bottleneck)
        self.init_module(self.ocr_distribute_head)
        self.init_module(self.ocr_gather_head)
        self.init_module(self.cls_head)


class SpatialGatherModule(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1).contiguous()  # batch x hw x c
        probs = torch.nn.functional.softmax(self.scale * probs, dim=2)  # N x K x HW
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).contiguous().unsqueeze(3)  # N x K x C
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            ConvBNRelu(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                       padding=0),
            ConvBNRelu(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                       padding=0),
        )
        self.f_object = nn.Sequential(
            ConvBNRelu(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                       padding=0),
            ConvBNRelu(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                       padding=0),
        )
        self.f_down = ConvBNRelu(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1,
                                 padding=0)
        self.f_up = ConvBNRelu(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                               padding=0)

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = nn.functional.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, x.size(2), x.size(3))
        context = self.f_up(context)
        if self.scale > 1:
            context = nn.functional.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=False)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock2D, self).__init__(in_channels, key_channels, scale)


class SpatialOCRModule(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1):
        super(SpatialOCRModule, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            ConvBNRelu(in_channels=_in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return output


def get_pose_net(cfg):
    model = HigherResolutionNet(cfg)
    model.init_weights('', verbose=False)
    return model
