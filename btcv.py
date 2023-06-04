import os
import os.path as osp
import random
import math
import sys

import torch
from torch.utils import data
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose

# 0: background
BTCV_label_num = {
    'spleen': 1,
    'right kidney': 2,
    'left kidney': 3,
    'gallbladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'inferior vena cava*': 9,
    'portal vein and splenic vein*': 10,
    'pancreas': 11,
    'right adrenal gland': 12,
    'left adrenal gland': 13
}

valid_dataset = {
    'background': 0,
    'liver': 1,
    'kidney': 2,
    'spleen': 3,
    'pancreas': 4,
}


class BTCVDataSet(data.Dataset):
    def __init__(self, root, split='test', crop_size=(64, 256, 256), mean=(128, 128, 128), scale=True,
                 ignore_label=255, is_transform=False):
        self.root = root
        self.split = split  # test: BTCV, val: FLARE21
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_transform = is_transform
        self.void_classes = [4, 5, 7, 8, 9, 10, 12, 13]

        print("Start preprocessing....")
        if self.split == 'test':
            # load data
            image_path = osp.join(self.root, 'img')
            label_path = osp.join(self.root, 'label')

            img_list = os.listdir(image_path)
            all_files = []
            for i, item in enumerate(img_list):
                img_file = osp.join(image_path, item)
                label_item = item.replace('img', 'label')
                label_file = osp.join(label_path, label_item)

                all_files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": item
                })

            self.files = all_files

        if self.split == 'val':
            # load data
            image_path = osp.join(self.root, 'TrainingImg')
            label_path = osp.join(self.root, 'TrainingMask')

            img_list = os.listdir(image_path)
            all_files = []
            for i, item in enumerate(img_list):
                img_file = osp.join(image_path, item)
                label_item = item.replace('_0000', '')
                label_file = osp.join(label_path, label_item)

                all_files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": item})

            # split train/val set
            train_X, val_X = train_test_split(all_files, test_size=0.20, shuffle=True, random_state=0)
            if self.split == 'train':
                self.files = train_X
            elif self.split == 'val':
                self.files = val_X

        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.

        # truncate
        CT = np.clip(CT, min_HU, max_HU)  # np.clip: 최대 최소값 제한
        # CT[np.where(CT <= min_HU)] = min_HU
        # CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])
        image = imageNII.get_fdata()
        label = labelNII.get_fdata()
        name = datafiles["name"]

        image = self.pad_image(image, [self.crop_h, self.crop_w, self.crop_d])
        label = self.pad_image(label, [self.crop_h, self.crop_w, self.crop_d])

        image = self.truncate(image)
        label = self.id2trainId(label)

        image = image[np.newaxis, :]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))

        label = self.extend_channel_classes(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label, name, labelNII.affine

    def id2trainId(self, label):
        if self.split == 'test':
            # void_class to background
            for void_class in self.void_classes:
                label[np.where(label == void_class)] = 0

            # left kidney to kidney(2)
            label[np.where(label == 3)] = 2
            # spleen to 3
            label[np.where(label == 1)] = 3
            # liver to 1
            label[np.where(label == 6)] = 1
            # pancreas to 4
            label[np.where(label == 11)] = 4

        shape = label.shape
        results_map = np.zeros((1, shape[0], shape[1], shape[2])).astype(np.float32)
        results_map[0, :, :, :] = label
        # results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        return results_map

    def extend_channel_classes(self, label):
        label_list = []
        for i in range(5):
            label_i = label.copy()
            label_i[label == i] = 1
            label_i[label != i] = 0
            label_list.append(label_i)
        stacked_label = np.stack(label_list, axis=1)
        stacked_label = np.squeeze(stacked_label)
        return stacked_label


if __name__ == '__main__':
    btcv = BTCVDataSet(root='./dataset/BTCV/Trainset', split='test')
    img_, label_, name_, label_aff = btcv[0]
    print("affine's type: {}".format(type(label_aff)))
    # flare = BTCVDataSet(root='./dataset/FLARE21', split='val')
    # valid_loader = data.DataLoader(dataset=flare, batch_size=1, shuffle=False, num_workers=0)
    # for train_iter, pack in enumerate(valid_loader):
    #     img_ = pack[0]
    #     label_ = pack[1]
    #     name_ = pack[2]
    #     label_affine = pack[3]
    #     print("img_shape: {}\nlabel_shape: {}\naffine's type: {}".format(img_.shape, label_.shape, type(label_affine)))
