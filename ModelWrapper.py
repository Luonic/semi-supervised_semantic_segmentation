import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeWrapper(nn.Module):
    def __init__(self, model, larger_side_size=1024):
        super(ResizeWrapper, self).__init__()
        self.model = model
        self.larger_side_size = larger_side_size
        self.sizes = torch.tensor([256, 512, 1024, 2048])

    def forward(self, input: torch.Tensor):
        larger_side_input_size = torch.max(input.size()[2:4])
        resize_ratio = (float(self.larger_side_size) / larger_side_input_size)
        smaller_side_size = min(input.size()[2:4])
        smaller_side_size_tgt = smaller_side_size * resize_ratio



        if input.size(2) > input.size(3):
            tgt_size = (self.larger_side_size, smaller_side_size_tgt)
        else:
            tgt_size = (smaller_side_size_tgt, self.larger_side_size)

        input = torch.nn.functional.interpolate(input, size=tgt_size, mode='bilinear')

        valid_tgt_sizes_mask = self.sizes > smaller_side_size_tgt
        valid_sizes = self.sizes[valid_tgt_sizes_mask]
        tgt_smaller_side_size = torch.min(valid_sizes)

        diff = (tgt_smaller_side_size - input.size(2)).to(torch.float32)


        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

        if input.size(2) > input.size(3):
            pad_left = torch.floor(diff / 2).to(torch.int32)
            pad_right = torch.ceil(diff / 2).to(torch.int32)
            pad_top = 0
            pad_bottom = 0
        else:
            pad_left = 0
            pad_right = 0
            pad_top = torch.floor(diff / 2).to(torch.int32)
            pad_bottom = torch.ceil(diff / 2).to(torch.int32)

        pad = (pad_left, pad_right, pad_top, pad_bottom)
        input = torch.nn.functional.pad(input, pad=pad)

        result =