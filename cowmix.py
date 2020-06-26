import math

import torch


def generate_gaussian(window_size, sigma):
    x = torch.arange(-window_size // 2, window_size // 2).float()
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def gaussian_kernel_2d_vertical(size, sigmas):
    gaussians = []
    for i in range(sigmas.shape[0]):
        gaussian = generate_gaussian(size, sigma=sigmas[i])
        gaussians.append(gaussian)
    gaussians = torch.stack(gaussians, dim=0)
    # gaussians = torch.unsqueeze()
    # tgt shape of weight: out_channels,1,kH,kW
    gaussians = torch.unsqueeze(gaussians, dim=1)
    gaussians = torch.unsqueeze(gaussians, dim=3)
    return gaussians


def dual_pass_gaussian_fileter2d(input, sigmas):
    # input: 1xNxHxW
    assert input.shape[1] == sigmas.shape[0]
    size = int(round(sigmas.max().item() * 3) * 2) + 1
    y_kernel = gaussian_kernel_2d_vertical(size, sigmas).to(input)
    x_kernel = torch.transpose(y_kernel, dim0=2, dim1=3)
    output = torch.nn.functional.conv2d(input, weight=y_kernel, padding=(math.floor(float(size) / 2), 0),
                                        groups=sigmas.shape[0])
    output = torch.nn.functional.conv2d(output, weight=x_kernel, padding=(0, math.floor(float(size) / 2)),
                                        groups=sigmas.shape[0])
    return output


def generate_cowmix_masks_like(example_tensor, mask_proportion_range, sigma_range):
    # mask_proportion range: tuple of 2 python floats
    # sigma_range: tuple of 2 python floats
    with torch.no_grad():
        p_distribution = torch.distributions.Uniform(torch.tensor(mask_proportion_range[0]),
                                                     torch.tensor(mask_proportion_range[1]))
        p = p_distribution.rsample(sample_shape=[example_tensor.size(0)])

        sigma_distribution = torch.distributions.Uniform(torch.tensor(math.log(float(sigma_range[0]))),
                                                         torch.tensor(math.log(float(sigma_range[1]))))

        sigmas = torch.exp(sigma_distribution.rsample([example_tensor.size(0)]))

        size = list(example_tensor.size())
        size[1] = 1
        mask = torch.normal(mean=0, std=1, size=size, dtype=example_tensor.dtype, device=example_tensor.device)
        mask = torch.transpose(mask, 0, 1)
        mask = dual_pass_gaussian_fileter2d(mask, sigmas)
        mask = torch.transpose(mask, 1, 0)

        mean = mask.mean(dim=(1, 2, 3), keepdim=True)
        std = mask.std(dim=(1, 2, 3), keepdim=True)

        # Compute threshold factors
        threshold_factors = (torch.erfinv(2 * p - 1) * math.sqrt(2.0)).to(example_tensor)
        threshold_factors = threshold_factors.reshape_as(mean)
        threshold_factors = threshold_factors * std + mean

        mask = (mask > threshold_factors).to(mask)
        return mask


def mix_with_mask(tensor_a, tensor_b, mask):
    return tensor_a * mask + tensor_b * (1. - mask)


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time

    while True:
        images = torch.rand(16, 3, 512, 512)
        start = time.time()
        mask = generate_cowmix_masks_like(images.to('cuda:0'), mask_proportion_range=(0.4, 0.6), sigma_range=(4, 16))
        print('took', time.time() - start, 'sec')

        mask = np.transpose(mask[0].cpu().numpy(), axes=(1, 2, 0))
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
