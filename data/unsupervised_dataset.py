import torch
from glob import glob
import numpy as np
import os
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class UnsupervisedImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, augmentations=None):
        self.filenames = glob(os.path.join(dataset_dir, '*.jp*g'))
        self.augmentations = augmentations

    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.filenames[item], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        augmented = self.augmentations(image=image)
        image = augmented['image']

        # image = norm(image)
        image = np.transpose(image, axes=(2, 0, 1))
        image = torch.from_numpy(image)

        sample = {'image': image}
        return sample

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    import configs.default_config as dcfg

    augmentations = dcfg.train['unsupervised_augmentations']
    dataset = UnsupervisedImagesDataset('/mnt/minio-pool/images', augmentations=augmentations)

    dataset = iter(dataset)
    for sample in dataset:
        image = sample['image']

        numpy_image = np.transpose(image.cpu().numpy(), axes=(1, 2, 0))
        print(numpy_image)
        cv2.imshow('image', cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)