from functools import partial
import torch

import optimizers.ranger

import data.dataset as dataset
import data.unsupervised_dataset as unsupervised_dataset

from models.deeplabv3 import deeplabv3_resnet101
from models.encoders import mobilenetv2
from models.unet import UNet

from models.discriminator import Discriminator

from albumentations import (
    RandomBrightnessContrast,
    RandomGamma,
    ToGray,
    ToFloat,
    Resize,
    Crop,
    CropNonEmptyMaskIfExists,
    HorizontalFlip,
    GridDistortion,
    LongestMaxSize,
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
    output_dir='runs/61',
    num_classes=2,
    image_size=1024,
)

# Model params
model = dict(model_fn=deeplabv3_resnet101,
             discriminator=partial(Discriminator,
                                   num_layers=5,
                                   initial_channels=64,
                                   max_depth=512,
                                   out_channels=1)
             )
# model = dict(model_fn=partial(UNet, encoder=mobilenetv2.mobilenet_v2(pretrained=True), max_width=128,
#                               norm_layer=torch.nn.BatchNorm2d, train_upsampling=False),
#              discriminator=partial(Discriminator,
#                                    num_layers=5,
#                                    initial_channels=64,
#                                    max_depth=512,
#                                    out_channels=1))

# Train params
train = dict(
    print_freq=10,
    batch_size_per_worker=2,
    num_dataloader_workers=4,
    base_lr=0.000025,
    crop_size=513,
    gradient_clip_value=10.0,

    supervised_adversarial_loss_weight=0.01, # 0.01
    unsupervised_adversarial_loss_weight=0.001, # 0.001
    semi_supervised_loss_weight=1.0, # 0.1
    semi_supervised_threshold=0.1,
    semi_supervised_training_start_step=5000
)
train['min_lr'] = train['base_lr'] * 0.0001
train['optimizer'] = partial(torch.optim.SGD,
                             lr=train['base_lr'],
                             momentum=0.9,
                             weight_decay=0.0005)
train['discriminator_optimizer'] = partial(torch.optim.Adam,
                                           lr=0.00001,
                                           betas=(0.9, 0.99))
train['lr_scheduler'] = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                mode='min',
                                factor=0.1,
                                patience=15,
                                verbose=False,
                                threshold=0.0001,
                                threshold_mode='rel',
                                cooldown=0,
                                min_lr=train['min_lr'],
                                eps=1e-08)
train['augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),
    PadIfNeeded(min_height=common['image_size'], min_width=common['image_size'], always_apply=True),
    Rotate(limit=45, always_apply=True),
    RandomResizedCrop(height=train['crop_size'], width=train['crop_size'], scale=(0.25, 1.0), ratio=(0.75, 1.33),
                      always_apply=True),
    HorizontalFlip(p=0.5),
    OneOf([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        RandomGamma(gamma_limit=(80, 120))
    ], p=1),
    ToGray(p=0.05),
    OneOf([
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0),
    ], p=1),
    OneOf([
        MotionBlur(blur_limit=7),
        GaussianBlur(blur_limit=7)
    ], p=0.1),
    OneOf([
        ImageCompression(quality_lower=70, quality_upper=90),
        ISONoise(color_shift=(0.01, 0.05),
                 intensity=(0.1, 0.5))
    ], p=0.1),
    # OneOf([
    #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #     GridDistortion(p=0.5),
    #     OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
    # ], p=0.8),
    ToFloat()
])
train['unsupervised_augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),
    PadIfNeeded(min_height=common['image_size'], min_width=common['image_size'], always_apply=True),
    RandomResizedCrop(height=train['crop_size'], width=train['crop_size'], scale=(0.25, 1.0), ratio=(0.75, 1.33),
                      always_apply=True),
    ToFloat()])
train['dataset'] = partial(dataset.SkinSegDataset,
                           dataset_dir='/home/alex/Code/instascraped/dataset_1',
                           augmentations=train['augmentations'],
                           partition=1)
train['unsupervised_dataset'] = partial(unsupervised_dataset.UnsupervisedImagesDataset,
                                        dataset_dir='/mnt/minio-pool/images',
                                        augmentations=train['unsupervised_augmentations'])

# Val params
val = dict(
    print_freq=10,
    batch_size_per_worker=1,
    num_dataloader_workers=1,
)
val['augmentations'] = ReplayCompose([
    LongestMaxSize(max_size=common['image_size'], always_apply=True),
    PadIfNeeded(min_height=common['image_size'], min_width=common['image_size'], always_apply=True),
    ToFloat()])
val['dataset'] = partial(dataset.SkinSegDataset,
                         dataset_dir='/home/alex/Code/instascraped/dataset_1',
                         augmentations=val['augmentations'],
                         partition=0)
