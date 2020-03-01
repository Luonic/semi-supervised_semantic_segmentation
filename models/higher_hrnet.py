# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import math

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

# pose_multi_resoluton_net related params
POSE_HIGHER_RESOLUTION_NET = CN()
POSE_HIGHER_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGHER_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGHER_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
POSE_HIGHER_RESOLUTION_NET.NUM_JOINTS = 2
POSE_HIGHER_RESOLUTION_NET.TAG_PER_JOINT = True

POSE_HIGHER_RESOLUTION_NET.STAGE1 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BRANCHES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BLOCKS = [4]
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_CHANNELS = [64]
POSE_HIGHER_RESOLUTION_NET.STAGE1.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE1.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [24, 48]
POSE_HIGHER_RESOLUTION_NET.STAGE2.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [24, 48, 92]
POSE_HIGHER_RESOLUTION_NET.STAGE3.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [24, 48, 92, 192]
POSE_HIGHER_RESOLUTION_NET.STAGE4.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.DECONV = CN()
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_DECONVS = 2
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_CHANNELS = [64, 32]
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_BASIC_BLOCKS = 4
POSE_HIGHER_RESOLUTION_NET.DECONV.KERNEL_SIZE = [4, 4]
POSE_HIGHER_RESOLUTION_NET.DECONV.CAT_OUTPUT = [True, True]

POSE_HIGHER_RESOLUTION_NET.LOSS = CN()
POSE_HIGHER_RESOLUTION_NET.LOSS.WITH_AE_LOSS = [False, False, False]

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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample if downsample is not None else torch.nn.Identity()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample is not None else torch.nn.Identity()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    # This module builds num_branches of conv layers for each resolution and
    # builds fuse layers to allow branches exchange information
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches: Final = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers: Final = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):

        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        # This funcrion builds list of parallel sequences of convolution blocks where each item in this list is
        # responsible for processing specific single scale
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # `i` is output branch idx
        # `j` is input branch idx
        # branch 0 is largest resolution
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsample smaller scale
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        # nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]):
        # if self.num_branches == 1:
        #     return [self.branches[0](x[0])]

        # for i in range(self.num_branches):
        #     x[i] = self.branches[i](x[i])

        # out = []
        branch_idx = 0
        for branch_module in self.branches:
            x[branch_idx] = branch_module(x[branch_idx])
            branch_idx += 1

        if self.num_branches == 1:
            return x

        x_fuse = []

        output_idx = 0
        for scale_fuse_layers in self.fuse_layers:
            tensors_to_fuse = []
            input_idx = 0
            for fuse_layer in scale_fuse_layers:
                resized_tensor = fuse_layer(x[input_idx])
                if resized_tensor.size() != x[output_idx].size():
                    resized_tensor = torch.nn.functional.interpolate(resized_tensor, size=x[output_idx].size()[2:4],
                                                                     mode='nearest')
                tensors_to_fuse.append(resized_tensor)
                input_idx += 1
            output_idx += 1

            y = torch.stack(tensors_to_fuse, dim=0).sum(dim=0)
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class Stage(nn.Module):
    __constants__ = ['mods']

    def __init__(self, layer_config, num_inchannels, multi_scale_output=True):
        super(Stage, self).__init__()
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            self.num_inchannels = modules[-1].get_num_inchannels()
            self.mods = nn.ModuleList(modules)

    def forward(self, input: List[torch.Tensor]):
        x = input
        for module in self.mods:
            x = module(x)
        return x


class Transition(nn.Module):
    # Transition module allows you to grow or reduce number of branches
    __constants__ = ['transition_layers']

    def __init__(self, num_channels_pre_layer, num_channels_cur_layer):
        super(Transition, self).__init__()
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        self.transition_layers = nn.ModuleList(transition_layers)

    def forward(self, input_list: List[torch.Tensor]):
        out = []
        x = input_list[0]
        branch_idx = 0
        for transition in self.transition_layers:
            y = transition(x)
            out.append(y)
            branch_idx += 1
            if branch_idx < len(input_list):
                x = input_list[branch_idx]
        return out


class HigherDecoderStage(nn.Module):
    # layers: Final
    # cat_output: Final

    def __init__(self, input_channels, output_channels, final_kernel_size, do_deconv, deconv_num_basic_blocks,
                 deconv_output_channels, deconv_kernel_size, cat_output):
        super(HigherDecoderStage, self).__init__()
        layers = []
        self.cat_output = cat_output

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
            features = torch.cat([features, output], dim=1)

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
            prev_feature_channels = deconv_output_channels[
                                        i] + output_channels if cat_output else deconv_output_channels

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, input: torch.Tensor):
        output: List[torch.Tensor] = []
        features = input
        for decoder_stage in self.decoder_layers:
            stage_output, features = decoder_stage(features)
            output.append(stage_output)
        return output


class PoseHigherResolutionNet(nn.Module):
    # stage2: Final[nn.Sequential]
    # stage3: Final[nn.Sequential]
    # stage4: Final[nn.Sequential]
    # transition1: Final[nn.ModuleList]
    # transition2: Final[nn.ModuleList]
    # transition3: Final[nn.ModuleList]

    __constants__ = ['stage2', 'stage3', 'stage4', 'transition1', 'transition2', 'transition3']

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # TODO: Replace hardcoded 256 with calculation of number of channels based on config
        # self.transition1 = self._make_transition_layer([256], num_channels)
        self.transition1 = Transition([256], num_channels)
        # self.stage2, pre_stage_channels = self._make_stage(
        #     self.stage2_cfg, num_channels)
        self.stage2 = Stage(self.stage2_cfg, num_channels)
        pre_stage_channels = self.stage2.num_inchannels

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # self.transition2 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        self.transition2 = Transition(pre_stage_channels, num_channels)
        # self.stage3, pre_stage_channels = self._make_stage(
        #     self.stage3_cfg, num_channels)
        self.stage3 = Stage(self.stage3_cfg, num_channels)
        pre_stage_channels = self.stage3.num_inchannels

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # self.transition3 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        self.transition3 = Transition(pre_stage_channels, num_channels)
        # self.stage4, pre_stage_channels = self._make_stage(
        #     self.stage4_cfg, num_channels, multi_scale_output=False)
        self.stage4 = Stage(self.stage4_cfg, num_channels, multi_scale_output=False)
        pre_stage_channels = self.stage4.num_inchannels

        # self.final_layers = self._make_final_layers(cfg, pre_stage_channels[0])
        # self.deconv_layers = self._make_deconv_layers(
        #     cfg, pre_stage_channels[0])

        self.decoder = HigherDecoder(input_channels=pre_stage_channels[0],
                                     output_channels=cfg.NUM_JOINTS,
                                     final_kernel_size=cfg.FINAL_CONV_KERNEL,
                                     num_deconvs=cfg.NUM_JOINTS,
                                     deconv_num_basic_blocks=cfg.DECONV.NUM_BASIC_BLOCKS,
                                     deconv_output_channels=cfg.DECONV.NUM_CHANNELS,
                                     deconv_kernel_size=cfg.DECONV.KERNEL_SIZE,
                                     cat_output=cfg.DECONV.CAT_OUTPUT)

        self.num_deconvs = cfg.DECONV.NUM_DECONVS
        self.deconv_config = cfg.DECONV
        self.loss_config = cfg.LOSS

        self.pretrained_layers = cfg['PRETRAINED_LAYERS']

    def _make_final_layers(self, cfg, input_channels):
        dim_tag = cfg.NUM_JOINTS if cfg.TAG_PER_JOINT else 1

        final_layers = []
        output_channels = cfg.NUM_JOINTS + dim_tag \
            if cfg.LOSS.WITH_AE_LOSS[0] else cfg.NUM_JOINTS
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0
        ))

        deconv_cfg = cfg.DECONV
        for i in range(deconv_cfg.NUM_DECONVS):
            input_channels = deconv_cfg.NUM_CHANNELS[i]
            output_channels = cfg.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i + 1] else cfg.NUM_JOINTS
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=cfg.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, cfg, input_channels):
        dim_tag = cfg.NUM_JOINTS if cfg.TAG_PER_JOINT else 1
        deconv_cfg = cfg.DECONV

        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):
            if deconv_cfg.CAT_OUTPUT[i]:
                final_output_channels = cfg.NUM_JOINTS + dim_tag \
                    if cfg.LOSS.WITH_AE_LOSS[i] else cfg.NUM_JOINTS
                input_channels += final_output_channels
            output_channels = deconv_cfg.NUM_CHANNELS[i]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            for _ in range(cfg.DECONV.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

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

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        # Transition layer allows you to grow or reduce number of branches

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list: List[torch.Tensor] = [x]

        x_list = self.transition1(y_list)
        # x_list = []
        # x = y_list[0]
        # branch_idx = 0
        # for transition in self.transition1:
        #     y = transition(x)
        #     x_list.append(y)
        #     branch_idx += 1
        #     if branch_idx < len(y_list):
        #         x = y_list[branch_idx]

        y_list: List[torch.Tensor] = self.stage2(x_list)

        x_list = self.transition2(y_list)
        # x_list = []
        # x = y_list[0]
        # branch_idx = 0
        # for transition in self.transition2:
        #     y = transition(x)
        #     x_list.append(y)
        #     branch_idx += 1
        #     if branch_idx < len(y_list):
        #         x = y_list[branch_idx]

        y_list = self.stage3(x_list)

        x_list = self.transition3(y_list)
        # x_list = []
        # x = y_list[0]
        # branch_idx = 0
        # for transition in self.transition3:
        #     y = transition(x)
        #     x_list.append(y)
        #     branch_idx += 1
        #     if branch_idx < len(y_list):
        #         x = y_list[branch_idx]

        y_list = self.stage4(x_list)

        # x = y_list[0]
        # y = self.final_layers[0](x)
        # final_outputs = []
        # final_outputs.append(y)
        #
        # for i in range(self.num_deconvs):
        #     if self.deconv_config.CAT_OUTPUT[i]:
        #         x = torch.cat((x, y), 1)
        #
        #     x = self.deconv_layers[i](x)
        #     y = self.final_layers[i + 1](x)
        #     final_outputs.append(y)

        x = y_list[0]
        # y = self.final_layers[0](x)
        # final_outputs = []
        # final_outputs.append(y)
        #
        # for i in range(self.num_deconvs):
        #     if self.deconv_config.CAT_OUTPUT[i]:
        #         x = torch.cat((x, y), 1)
        #
        #     x = self.deconv_layers[i](x)
        #     y = self.final_layers[i + 1](x)
        #     final_outputs.append(y)
        final_outputs = self.decoder(x)

        # for t in enumerate(final_outputs):
        #     print(t[0], t[1].size())
        return final_outputs

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


def get_pose_net(cfg, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)
    model.init_weights('', verbose=False)
    return model
