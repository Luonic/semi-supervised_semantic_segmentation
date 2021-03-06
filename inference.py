import config
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from models import inference_wrapper

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

cfg_path = 'configs/default_config.py'
cfg = config.fromfile(cfg_path)

device = 'cuda:0'
# device = 'cpu'
# model = cfg['model']['model_fn'](cfg['common']['num_classes'])
model = cfg['model']['model_fn']()
model.to('cpu')
model.eval()

workdir = cfg['common']['output_dir']
latest_checkpoint_path = os.path.join(workdir, 'best.pth')
# latest_checkpoint_path = os.path.join(workdir, 'checkpoint.pth')
checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

model.fuse_model()
model.num_scales = torch.tensor(3)

model = inference_wrapper.InferenceWrapper(model)

with torch.jit.optimized_execution(True):
    ts_model = torch.jit.script(model)
    torch.jit.save(ts_model, os.path.join(workdir, 'model.ts'))
    # exit(0)
    del ts_model
    # model = ts_model


model.to(device)

torch.backends.quantized.engine = 'fbgemm'
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# val_dataset = cfg['val']['dataset']()
from albumentations import ReplayCompose, LongestMaxSize, SmallestMaxSize, PadIfNeeded, ToFloat

augmentations = ReplayCompose([
    LongestMaxSize(max_size=cfg['common']['image_size'], always_apply=True),
    # LongestMaxSize(max_size=2048, always_apply=True),
    # PadIfNeeded(min_height=cfg['common']['image_size'], min_width=cfg['common']['image_size'], always_apply=True,
    #             border_mode=cv2.BORDER_CONSTANT),
    ToFloat()])
# augmentations = cfg['val']['augmentations']
val_dataset = cfg['train']['unsupervised_dataset'](augmentations=augmentations)
# val_dataset = cfg['val']['dataset'](augmentations=augmentations)
# val_dataset = cfg['train']['dataset'](augmentations=augmentations)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             pin_memory=False,
                                             drop_last=False,
                                             num_workers=1,
                                             shuffle=True)

with torch.no_grad(), torch.jit.optimized_execution(True):
    for i, sample in tqdm(enumerate(val_dataloader), smoothing=0.01):
        if i == 200:
            print('Quantizing...')
            torch.quantization.convert(model, inplace=True)
            with torch.jit.optimized_execution(True):
                model = torch.jit.script(model)
            torch.jit.save(model, os.path.join(workdir, 'model_quantized.ts'))
            exit(0)
        image = sample['image'].to(device)
        # mask = sample['semantic_mask'].to(device, non_blocking=True)

        image = image[0]
        binary_mask, prob_mask = model(image)

        print('max', prob_mask.max().item(), 'min', prob_mask.min().item())
        skin_mask = (prob_mask[1].cpu().detach().numpy() * 255).astype(np.uint8)

        numpy_image = cv2.cvtColor(np.transpose(image.cpu().numpy(), axes=(1, 2, 0)), cv2.COLOR_RGB2BGR)

        ratio = 1650 / 2 / max(numpy_image.shape)
        skin_mask = cv2.resize(skin_mask, None, fx=ratio * 1, fy=ratio)
        numpy_image = cv2.resize(numpy_image, None, fx=ratio, fy=ratio)

        cv2.imshow('skin', skin_mask)
        cv2.imshow('image', numpy_image)
        cv2.waitKey(32)

max_view_size = 700

# with torch.no_grad():
#     for sample in val_dataloader:
#         image = sample['image'].to(device)
#         mask = sample['semantic_mask'].to(device, non_blocking=True)
#
#         for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
#
#             # pred_map = model(image)['out']
#             image_resized = torch.nn.functional.interpolate(image, scale_factor=scale, align_corners=False, mode='bilinear')
#             pred_map = model(image_resized)
#             pred_map = torch.nn.functional.interpolate(pred_map, size=(image.size(2), image.size(3)), mode='bilinear')
#             skin_mask = pred_map[0, 1].cpu().numpy()
#
#             numpy_image = cv2.cvtColor(np.transpose(image[0].cpu().numpy(), axes=(1, 2, 0)), cv2.COLOR_RGB2BGR)
#             ratio = float(max_view_size) / max(numpy_image.shape)
#             numpy_image = cv2.resize(numpy_image, None, fx=ratio, fy=ratio)
#             skin_mask = cv2.resize(skin_mask, None, fx=ratio, fy=ratio)
#
#             cv2.imshow('skin', skin_mask)
#             cv2.imshow('image', numpy_image)
#             cv2.waitKey(10)

# import kornia.geometry.transform.affwarp as kaffine
# with torch.no_grad():
#     for sample in val_dataloader:
#         image = sample['image'].to(device)
#         mask = sample['semantic_mask'].to(device, non_blocking=True)
#
#         for angle in [-45, -22.5, 0, 22.5, 45]:
#
#             # pred_map = model(image)['out']
#             image_rotated = kaffine.rotate(image, torch.from_numpy(np.array(angle)).to(device))
#             pred_map = model(image_rotated)
#             pred_map = torch.nn.functional.interpolate(pred_map, size=(image.size(2), image.size(3)), mode='bilinear')
#             skin_mask = pred_map
#             skin_mask = kaffine.rotate(skin_mask, torch.from_numpy(np.array(-angle)).to(device))
#             skin_mask = skin_mask[0, 1].cpu().numpy()
#
#             numpy_image = cv2.cvtColor(np.transpose(image[0].cpu().numpy(), axes=(1, 2, 0)), cv2.COLOR_RGB2BGR)
#
#             cv2.imshow('skin', skin_mask)
#             cv2.imshow('image', numpy_image)
#             cv2.waitKey(10)
