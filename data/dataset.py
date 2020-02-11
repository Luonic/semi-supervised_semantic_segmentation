import torch
import torch.nn.functional as F
from glob import glob
import os
import torchvision
import cv2
import numpy as np
import json
from albumentations import ReplayCompose

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def torch_resize(tensor, size, mode):
    batch = torch.unsqueeze(tensor, dim=0)
    if mode == 'bilinear':
        align_corners = False
    else:
        align_corners = None
    batch = torch.nn.functional.interpolate(batch, size=size, mode=mode, align_corners=align_corners)
    return torch.squeeze(batch, dim=0)


class SkinSegDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, augmentations=None, partition=None):
        self.class2idx = {'background': 0, 'skin': 1}
        if partition is None:
            raise ValueError('Partition should be integer and not None')
        filenames_whitelist = []

        with open(os.path.join(dataset_dir, 'part_config.json')) as part_f:
            partitions = json.load(part_f)

        if isinstance(partition, int):
            filenames_whitelist.extend(partitions[partition])
        elif isinstance(partition, (list, tuple)):
            for part_idx in partition:
                filenames_whitelist.extend(partitions[part_idx])
        del partitions

        filenames = glob(os.path.join(dataset_dir, '*', 'image.jpg'))

        if filenames_whitelist:
            filenames_whitelist = set(filenames_whitelist)
            filenames = [filename for filename in filenames if
                         os.path.splitext(os.path.basename(os.path.dirname(filename)))[0] in filenames_whitelist]

        self.filenames = filenames #* 20
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.filenames[item], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask_filenames = glob(os.path.join(os.path.dirname(self.filenames[item]), '*.png'))
        semantic_mask = np.zeros(shape=(len(self.class2idx.keys()), image.shape[0], image.shape[1]), dtype=np.float32)
        for file_path in mask_filenames:
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask /= 255.
            class_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
            class_idx = self.class2idx[class_name]
            semantic_mask[class_idx] += mask

        semantic_mask = np.clip(semantic_mask, 0., 1.)
        semantic_mask[0] = np.ones_like(semantic_mask[0]) - np.max(semantic_mask, axis=0, keepdims=False)
        semantic_mask /= np.sum(semantic_mask, axis=0, keepdims=True)
        semantic_mask = (semantic_mask * 255).astype(np.uint8)
        semantic_mask = np.transpose(semantic_mask, axes=(1, 2, 0))

        # Augmentation
        augmented = self.augmentations(image=image, mask=semantic_mask)
        image = augmented['image']
        semantic_mask = augmented['mask']
        # instance_mask = ReplayCompose.replay(augmented['replay'], mask=instance_mask)['mask']

        # image = norm(image)
        image = np.transpose(image, axes=(2, 0, 1))
        image = torch.from_numpy(image)

        semantic_mask = np.transpose(semantic_mask, axes=(2, 0, 1))
        semantic_mask = semantic_mask.astype(np.float32) / 255
        semantic_mask = torch.from_numpy(semantic_mask)

        sample = {'image': image,
                  'semantic_mask': semantic_mask}
        return sample

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    import configs.default_config as dcfg

    augmentations = dcfg.train['augmentations']
    dataset = SkinSegDataset(dataset_dir='/home/alex/Code/instascraped/dataset_1',
                             augmentations=augmentations,
                             partition=1)
    for sample in dataset:
        image = sample['image']
        mask = sample['semantic_mask']

        numpy_image = np.transpose(image.cpu().numpy(), axes=(1, 2, 0))
        numpy_mask = np.transpose(mask.cpu().numpy(), axes=(1, 2, 0))

        cv2.imshow('skin', numpy_mask[:, :, 1])
        cv2.imshow('image', cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))
        print('mask', mask)
        print('image', image)
        cv2.waitKey(0)