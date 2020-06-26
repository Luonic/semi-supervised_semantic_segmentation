import torch
import torch.nn as nn
import torch.nn.functional as F
from models.higher_hrnet import ConvBN, ConvBNRelu
from typing import Tuple, List

class MultiscaleAttention(nn.Module):
    # Implements "HIERARCHICAL MULTI-SCALE ATTENTION FOR SEMANTIC SEGMENTATION"
    # https://arxiv.org/abs/2005.10821
    def __init__(self, model_fn, num_feature_channels, num_scales):
        super(MultiscaleAttention, self).__init__()
        self.model = model_fn()
        self.attention_head = nn.Sequential(
            ConvBNRelu(num_feature_channels, num_feature_channels // 2, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(num_feature_channels // 2, num_feature_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(num_feature_channels // 4, 1, kernel_size=1, stride=1, padding=0),
        )
        # self.register_buffer('num_scales', torch.tensor(num_scales))
        self.num_scales = num_scales
        self.float_add = torch.nn.quantized.FloatFunctional()
        self.float_mul_low_res = torch.nn.quantized.FloatFunctional()
        self.float_mul_high_res = torch.nn.quantized.FloatFunctional()

        # self.quant = torch.quantization.QuantStub()
        # self.features_quant = torch.quantization.QuantStub()
        self.dequant_att_mask = torch.quantization.DeQuantStub()

    def downsample(self, input, factor):
        return F.interpolate(input,
                             (input.size(2) // factor,
                              input.size(3) // factor),
                             mode='bilinear',
                             align_corners=False)

    def upsample_to(self, input, target_size: List[int]):
        return F.interpolate(input, target_size, mode='bilinear', align_corners=False)

    def forward(self, input: torch.Tensor):
        # input = self.quant(input)
        features = []
        features_and_logits = self.model(self.downsample(input, factor=2 ** (self.num_scales - 1)))
        low_res_features = features_and_logits[0][0]
        low_res_logits = features_and_logits[1][0]
        features.append(low_res_features)
        out_logits = low_res_logits
        scale_idx = self.num_scales - 2
        while scale_idx >= 0:
            features_and_logits = self.model(self.downsample(input, factor=2 ** scale_idx))
            high_res_features = features_and_logits[0][0]
            high_res_logits = features_and_logits[1][0]
            attention_mask = self.attention_head(low_res_features)
            attention_mask = torch.sigmoid(self.dequant_att_mask(self.upsample_to(attention_mask, high_res_logits.size()[2:4])))
            out_logits = self.upsample_to(out_logits, high_res_logits.size()[2:4])
            out_logits = out_logits * attention_mask + high_res_logits * (1 - attention_mask)
            low_res_features = high_res_features
            features.append(low_res_features)
            scale_idx -= 1
        return features, [out_logits]

    def fuse_model(self):
        def fuse_fn(m):
            if hasattr(m, 'fuse') and callable(m.fuse):
                m.fuse()

        self.apply(fuse_fn)


