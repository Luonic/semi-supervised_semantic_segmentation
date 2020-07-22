# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from functools import partial
import torch
import losses
import cv2
from optimizers.ranger import Ranger
from optimizers.diffmod import DiffMod

import data.dataset as dataset
import data.unsupervised_dataset as unsupervised_dataset

from models.deeplabv3 import deeplabv3_resnet101
from models.deeplabv3 import deeplabv3_resnet50
from models.encoders import mobilenetv2
from models.unet import UNet
from models.simple_unet import UNet as SimpleUNet
from models.hardnet import HarDNet
from models.deeplabv3 import fcn_resnet50
from models.higher_hrnet import get_pose_net, POSE_HIGHER_RESOLUTION_NET
from models.multiscale_attention import MultiscaleAttention

from models.discriminator import Discriminator

from albumentations import (
    RandomBrightnessContrast,
    RandomGamma,
    ToGray,
    ToFloat,
    Resize,
    Crop,
    CenterCrop,
    CropNonEmptyMaskIfExists,
    HorizontalFlip,
    GridDistortion,
    LongestMaxSize,
    SmallestMaxSize,
    PadIfNeeded,
    RandomResizedCrop,
    ShiftScaleRotate,
    Rotate,
    ElasticTransform,
    ImageCompression,
    MotionBlur,
    RGBShift,
    HueSaturationValue,
    OpticalDistortion,
    GaussianBlur,
    ISONoise,
    ReplayCompose,
    OneOf,
)

# Common params
common = dict(
    world_size=4,
    use_cpu=False,
    workers=4,
    output_dir='runs/55_ema_hrnet-small-msa-classic-transfuse-w48_Focal0.5-RMISigmoid0.5_SGDR-To10-Tm2_harder-aug_finetune_dataset_5',
    num_classes=2,
    image_size=1024,
)

# Model params
# model = dict(model_fn=deeplabv3_resnet101,
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=partial(UNet, encoder=mobilenetv2.mobilenet_v2(pretrained=True), max_width=128,
#                               norm_layer=torch.nn.BatchNorm2d, train_upsampling=True),
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=partial(SimpleUNet, num_blocks=7, first_channels=32, max_width=256, norm_layer=torch.nn.BatchNorm2d, train_upsampling=True),
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=partial(HarDNet),
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=deeplabv3_resnet50,
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=fcn_resnet50,
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# model = dict(model_fn=partial(get_pose_net, cfg=POSE_HIGHER_RESOLUTION_NET),
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))
model = dict(model_fn=partial(MultiscaleAttention, model_fn=partial(get_pose_net, cfg=POSE_HIGHER_RESOLUTION_NET), num_feature_channels=32+64+128+256, num_scales=2),
             discriminator=partial(Discriminator,
                                   num_layers=5,
                                   initial_channels=64,
                                   max_depth=512,
                                   out_channels=1))

# Train params
train = dict(
    print_freq=9,
    batch_size_per_worker=10,
    virtual_batch_size_multiplier=1,
    num_dataloader_workers=4,

    crop_size=512,
    gradient_clip_value=5.0,

    # Semi-supervised training hyperparams
    use_semi_supervised=False,
    mask_proportion_range=(0.45, 0.55),
    sigma_range=(8, 32),
    consistency_loss_weight=10,
    ema_model_alpha=0.99,
    confidence_threshold=0.97
)
train['base_lr'] = 0.0001 * train['virtual_batch_size_multiplier'] / 4 * 9
train['loss'] = losses.CalculateLoss([
    # {'loss_fn': losses.DenseCrossEntropyLossWithLogits(reduction='mean'), 'weight': [0.5]},
    {'loss_fn': losses.DenseBinaryCrossEntropyLossWithLogits(reduction='mean'), 'weight': [0.5]},
    # {'loss_fn': losses.DiceWithLogitsLoss(), 'weight': [0.125]},
    # {'loss_fn': losses.OhemCrossEntropy(), 'weight': [1.0]}
    # {'loss_fn': losses.FocalLoss(alpha=0.5, gamma=2), 'weight': [0.5]},
    {'loss_fn': losses.RMILoss(num_classes=2, rmi_radius=3, rmi_pool='avg', rmi_pool_size=4, rmi_pool_stride=4), 'weight': [0.5]},
])

train['min_lr'] = train['base_lr'] * 0.001
train['optimizer'] = partial(torch.optim.SGD,
                             lr=train['base_lr'],
                             momentum=0.9,
                             weight_decay=0.0005)
# train['optimizer'] = partial(torch.optim.Adam,
#                              lr=train['base_lr'])
# train['optimizer'] = partial(DiffMod,
#                              lr=train['base_lr'],
#                              betas=(0.9, 0.999),
#                              len_memory=1000,
#                              version=0,
#                              eps=1e-8,
#                              weight_decay=0.0001)
# train['lr_scheduler'] = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
#                                 mode='min',
#                                 factor=0.1,
#                                 patience=20, # 20
#                                 verbose=False,
#                                 threshold=0.0001,
#                                 threshold_mode='rel',
#                                 cooldown=0,
#                                 min_lr=train['min_lr'],
#                                 eps=1e-08)
train['lr_scheduler'] = partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                T_0=300,
                                T_mult=2,
                                eta_min=train['base_lr'] * 0.01,
                                last_epoch=-1)
train['augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),
    # SmallestMaxSize(max_size=common['image_size'], always_apply=True),
    PadIfNeeded(min_height=train['crop_size'], min_width=train['crop_size'], always_apply=True,
                border_mode=cv2.BORDER_CONSTANT),
    Rotate(limit=15, always_apply=True),
    RandomResizedCrop(height=train['crop_size'], width=train['crop_size'], scale=(0.25, 1.0), ratio=(0.75, 1.33),
                      always_apply=True),
    # Resize(height=train['crop_size'], width=train['crop_size'], always_apply=True),
    HorizontalFlip(p=0.5),
    OneOf([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        # RandomGamma(gamma_limit=(80, 120))
    ], p=1),
    ToGray(p=0.1),
    OneOf([
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=1),
    ], p=0.3),
    OneOf([
    # #     #######MotionBlur(blur_limit=7), # Does not work correctly
        GaussianBlur(blur_limit=9)
    ], p=0.1),
    OneOf([
    #     ImageCompression(quality_lower=70, quality_upper=90),
        ISONoise(color_shift=(0.01, 0.05),
                 intensity=(0.1, 0.5))
    ], p=0.2),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
    ], p=0.5),
    ToFloat()
])
train['unsupervised_augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),
    PadIfNeeded(min_height=train['crop_size'], min_width=train['crop_size'], always_apply=True,
                border_mode=cv2.BORDER_CONSTANT),
    RandomResizedCrop(height=train['crop_size'], width=train['crop_size'], scale=(0.25, 1.0), ratio=(1., 1.),
                      always_apply=True),
    HorizontalFlip(p=0.5),
    ToFloat()])
train['dataset'] = partial(dataset.SkinSegDataset,
                           # dataset_dir='/home/alex/Code/instascraped/dataset_coco_no-blank',
                           # dataset_dir='/mnt/ramdisk/dataset_coco_no-blank',
                           dataset_dir='/home/alex/Code/instascraped/dataset_5_no-blank',
                           augmentations=train['augmentations'],
                           partition=1)
train['unsupervised_dataset'] = partial(unsupervised_dataset.UnsupervisedImagesDataset,
                                        dataset_dirs=[
                                            '/home/alex/Code/instascraped/annotations_partitions/unsupervised_partition_1',
                                            '/home/alex/Code/instascraped/annotations_partitions/unsupervised_partition_2',
                                            '/home/alex/Code/instascraped/annotations_partitions/unsupervised_partition_3'
                                        ],
                                        augmentations=train['unsupervised_augmentations'])
train['pretrained_checkpoint_path'] = 'runs/44_hrnet-small-msa-classic-transfuse-w48_BCE0.5-RMISigmoid0.5_SGDR-To10-Tm2_harder-aug_coco-no-blank-pretrain/checkpoint.pth'
# train['pretrained_checkpoint_path'] = ''

# Val params
val = dict(
    print_freq=10,
    batch_size_per_worker=1,
    num_dataloader_workers=4,
)
val['augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),

    # SmallestMaxSize(max_size=common['image_size'], always_apply=True),
    # CenterCrop(height=common['image_size'], width=common['image_size'], always_apply=True),

    # PadIfNeeded(min_height=common['image_size'], min_width=common['image_size'], always_apply=True, border_mode=cv2.BORDER_CONSTANT),
    # PadIfNeeded(min_height=train['crop_size'], min_width=train['crop_size'], border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ToFloat()])
val['dataset'] = partial(dataset.SkinSegDataset,
                         # dataset_dir='/home/alex/Code/instascraped/dataset_coco_no-blank',
                         # dataset_dir='/mnt/ramdisk/dataset_coco_no-blank',
                         dataset_dir='/home/alex/Code/instascraped/dataset_5_no-blank',
                         augmentations=val['augmentations'],
                         partition=0)
