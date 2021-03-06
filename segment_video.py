import cv2
import numpy as np

import torch
from albumentations import Compose, LongestMaxSize, ToFloat, MotionBlur

MODEL_PATH = 'runs/55_ema_hrnet-small-msa-classic-transfuse-w48_Focal0.5-RMISigmoid0.5_SGDR-To10-Tm2_harder-aug_finetune_dataset_5/model.ts'
MODEL_DEVICE = 'cuda:1'
# MODEL_DEVICE = 'cpu'
model = torch.jit.load(MODEL_PATH)
model.to(MODEL_DEVICE)
augmentations = Compose([
    LongestMaxSize(max_size=1024, always_apply=True),
    # MotionBlur(blur_limit=25, always_apply=True),
    # PadIfNeeded(min_height=1024, min_width=1024, always_apply=True),
    ToFloat()])

video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
# video_capture = cv2.VideoCapture('/home/alex/Videos/nike_ad.mp4')

while True:
    # Capture frame-by-frame
    for i in range(1):
        ret, image = video_capture.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image = augmentations(image=image)['image']
    image = np.transpose(image, axes=(2, 0, 1))
    with torch.no_grad(), torch.jit.optimized_execution(True):
        image = torch.from_numpy(image).to(MODEL_DEVICE)
        binary_mask, prob_mask = model(image)
        mask = prob_mask[1:]
        binary_mask = binary_mask[1:]
        # binary_mask = binary_mask * mask
        binary_mask = np.transpose(binary_mask.cpu().detach().numpy(), axes=(1, 2, 0))
        mask = np.transpose(mask.cpu().detach().numpy(), axes=(1, 2, 0))
    red_tensor = np.array([[[1, 0, 0]]], dtype=np.float32)
    visualization_mask = binary_mask * red_tensor

    resize_ratio = 800 / max(orig_image.shape)
    image = cv2.cvtColor(cv2.resize(orig_image, None, fx=resize_ratio, fy=resize_ratio), cv2.COLOR_RGB2BGR)
    visualization_mask = cv2.resize(visualization_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    binary_mask = np.expand_dims(cv2.resize(binary_mask, (image.shape[1], image.shape[0])), axis=2)
    image = image * (1 - binary_mask) + (image * (0.5 * binary_mask) + visualization_mask * (0.5 * binary_mask))
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.waitKey(1)
