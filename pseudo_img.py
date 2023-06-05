import os
import os.path as osp
import sys
import random
import math

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
import nibabel as nib
from skimage.transform import resize
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose


class FAKEDataSet(data.Dataset):
    def __init__(self, root, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, split='train'):
        self.root = root
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label
        self.split = split
        self.files = []

        print("Start preprocessing....")
        # load data
        image_path = osp.join(self.root, 'Img')
        label_path = osp.join(self.root, 'Pred')

        img_list = os.listdir(image_path)
        all_files = []
        for i, item in enumerate(img_list):
            img_file = osp.join(image_path, item)
            label_item = item.replace('img', 'pred')
            label_file = osp.join(label_path, label_item)

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item
            })

        # split train/val set
        train_X, val_X = train_test_split(all_files, test_size=0.20, shuffle=True, random_state=0)
        if self.split == 'train':
            self.files = train_X
        else:
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
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((1, shape[0], shape[1], shape[2])).astype(np.float32)
        results_map[0, :, :, :] = label
        return results_map

    def pad_image(self, img, target_size):
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

        if self.scale and np.random.uniform() < 0.2:
            scaler = np.random.uniform(0.7, 1.4)
        else:
            scaler = 1

        image = self.pad_image(image, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])
        label = self.pad_image(label, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])

        image = self.truncate(image)
        label = self.id2trainId(label)

        image = image[np.newaxis, :]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Channel x Depth x H x W

        if self.is_mirror:
            if np.random.rand(1) <= 0.5:  # flip W
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            if np.random.rand(1) <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            if np.random.rand(1) <= 0.5:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]

        d, h, w = image.shape[-3:]
        if scaler != 1 or d != self.crop_d or h != self.crop_h or w != self.crop_w:
            image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (1, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True,
                           preserve_range=True)

        label = self.extend_channel_classes(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        if self.split == 'train':
            tr_transforms = get_train_transform()
            image = tr_transforms(image)
            label = tr_transforms(label)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label, name

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


def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


if __name__ == '__main__':
    img_path = './sample/Img/0img.nii.gz'
    pred_path = './sample/Pred/0pred.nii.gz'
    imageNII = nib.load(img_path)
    labelNII = nib.load(pred_path)
    image = imageNII.get_fdata()
    label = labelNII.get_fdata()
    print("label's type: {}".format(type(label)))
    print("img's shape: {}\nlabel's shape: {}".format(image.shape, label.shape))
    # print("img\n{}".format(image))

    d1, d2, d3 = image.shape
    max_score = 0
    max_score_idx = 0
    for i in range(d3):
        sagital_label = label[:, :, i]
        classes = np.unique(sagital_label)
        if classes.size >= 1:
            counts = np.array([max(np.where(sagital_label == c)[0].size, 1e-8) for c in range(5)])
            score = np.exp(np.sum(np.log(counts)) - 5 * np.log(np.sum(counts)))
            if score > max_score:
                max_score = score
                max_score_idx = i

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image[30, :, :], cmap='gray')
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label[:, :, max_score_idx])
    plt.title('Ground Truth')
    plt.show()

    # train_path = './sample'
    # train_data = FAKEDataSet(root=train_path, split='train')
    # train_loader = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=4)
    # for train_iter, pack in enumerate(train_loader):
    #     img_ = pack[0]
    #     label_ = pack[1]
    #     name_ = pack[2]
    #     print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))
