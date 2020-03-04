import cv2
import numpy as np

import torch
from albumentations import ReplayCompose, LongestMaxSize, ToFloat

MODEL_PATH = 'runs/36_bce_pose-hrnet_crop-512_size-1024_coco-full_pretrain/model.ts'
MODEL_DEVICE = 'cuda:1'
# MODEL_DEVICE = 'cpu'
model = torch.jit.load(MODEL_PATH)
model.to(MODEL_DEVICE)
augmentations = ReplayCompose([
    LongestMaxSize(max_size=1024, always_apply=True),
    # PadIfNeeded(min_height=1024, min_width=1024, always_apply=True),
    ToFloat()])

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, image = video_capture.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image = augmentations(image=image)['image']
    image = np.transpose(image, axes=(2, 0, 1))
    image_batch_np = np.expand_dims(image, axis=0)
    with torch.no_grad():
        image = torch.from_numpy(image_batch_np).to(MODEL_DEVICE)
        pred_maps = model(image)
        pred_map = torch.sigmoid(pred_maps[-1])
        mask = pred_map[0, 1:]
        binary_mask = (mask > 0.5).to(mask)
        binary_mask = np.transpose(binary_mask.cpu().detach().numpy(), axes=(1, 2, 0))
        mask = np.transpose(mask.cpu().detach().numpy(), axes=(1, 2, 0))
    red_tensor = np.array([[[1, 0, 0]]], dtype=np.float32)
    visualization_mask = binary_mask * red_tensor

    resize_ratio = 800 / max(orig_image.shape)
    image = cv2.cvtColor(cv2.resize(orig_image, None, fx=resize_ratio, fy=resize_ratio), cv2.COLOR_RGB2BGR)
    visualization_mask = cv2.resize(visualization_mask, (image.shape[1], image.shape[0]))
    binary_mask = np.expand_dims(cv2.resize(binary_mask, (image.shape[1], image.shape[0])), axis=2)
    image = image * (1 - binary_mask) + (image * (0.5 * binary_mask) + visualization_mask * (0.5 * binary_mask))
    mask = cv2.resize(mask, None, fx=resize_ratio, fy=resize_ratio)

    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.waitKey(1)
