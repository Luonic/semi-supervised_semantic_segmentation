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

train_augmentations = ReplayCompose([
    LongestMaxSize(max_size=2048, always_apply=True),
    Rotate(limit=45, always_apply=True),
    RandomResizedCrop(height=512, width=512, scale=(0.25, 1.0), ratio=(0.75, 1.33), always_apply=True),
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

val_augmentations = ReplayCompose([LongestMaxSize(max_size=1024, always_apply=True),
                             ToFloat()])
