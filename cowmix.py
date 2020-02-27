import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

def generate_cowmix_mask_like(example_tensor, sigma):
    with torch.no_grad():
        size = list(example_tensor.size())
        size[1] = 1
        mask = torch.normal(mean=0, std=1, size=size, dtype=example_tensor.dtype)
        mask = kornia.gaussian_blur2d(mask, kernel_size=(255, 255), sigma=(8, 8))
        threshold = 0.
        mask = (mask > threshold).to(mask)
        return mask


if __name__ == '__main__':
    images = torch.rand(1, 3, 512, 512)
    mask = generate_cowmix_mask_like(images, sigma=129)
    import cv2
    import numpy as np
    mask = np.transpose(mask[0].cpu().numpy(), axes=(1, 2, 0))
    cv2.imshow('mask', mask)
    cv2.waitKey(0)