import os
import os.path as osp
import sys
import random
import collections
import math
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose


class FLAREDataSet(data.Dataset):
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

        spacing = [0.8, 0.8, 1.5]

        print("Start preprocessing....")
        # load data
        image_path = osp.join(self.root, 'TrainingImg')
        label_path = osp.join(self.root, 'TrainingMask')

        img_list = os.listdir(image_path)
        all_files = []
        for i, item in enumerate(img_list):
            img_file = osp.join(image_path, item)
            label_item = item.replace('_0000', '')
            label_file = osp.join(label_path, label_item)

            label = nib.load(label_file).get_fdata()
            # if task_id == 1:
            #     label = label.transpose((1, 2, 0))
            boud_h, boud_w, boud_d = np.where(label >= 1)  # background 아닌

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item,
                "spacing": spacing,
                "bbx": [boud_h, boud_w, boud_d]
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

    def locate_bbx(self, label, scaler, bbx):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = bbx
        margin = 32  # pixels

        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            bbx_h_mint = bbx_h_min - math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            if bbx_h_mint < 0:
                bbx_h_maxt -= bbx_h_mint
                bbx_h_mint = 0
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            bbx_w_mint = bbx_w_min - math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            if bbx_w_mint < 0:
                bbx_w_maxt -= bbx_w_mint
                bbx_w_mint = 0
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            bbx_d_mint = bbx_d_min - math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            if bbx_d_mint < 0:
                bbx_d_maxt -= bbx_d_mint
                bbx_d_mint = 0
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        # if random.random() < 0.8:
        #     d0 = random.randint(bbx_d_min, np.max([bbx_d_max - scale_d, bbx_d_min]))
        #     h0 = random.randint(bbx_h_min, np.max([bbx_h_max - scale_h, bbx_h_min]))
        #     w0 = random.randint(bbx_w_min, np.max([bbx_w_max - scale_w, bbx_w_min]))
        # else:
        #     d0 = random.randint(0, img_d - scale_d)
        #     h0 = random.randint(0, img_h - scale_h)
        #     w0 = random.randint(0, img_w - scale_w)

        # more BG patch crop
        # d0 = random.randint(0, img_d - 20)
        # h0 = random.randint(0, img_h - 20)
        # w0 = random.randint(0, img_w - 20)

        # no patch outside the image
        d0 = random.randint(0, img_d - scale_d)
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

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

        if self.split == 'train':
            [h0, h1, w0, w1, d0, d1] = self.locate_bbx(label, scaler, datafiles["bbx"])
            image = image[h0: h1, w0: w1, d0: d1]
            label = label[h0: h1, w0: w1, d0: d1]

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


def my_collate(batch):
    image, label, name = zip(*batch)
    image = torch.stack(image, 0)
    label = torch.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'image': image,
                 'label': label,
                 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


if __name__ == '__main__':
    # flare = FLAREDataSet(root='./dataset/FLARE21', split='train')
    # img_, label_, name_ = flare[0]
    # print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))

    flare = FLAREDataSet(root='./dataset/FLARE21', split='train')
    train_loader = data.DataLoader(dataset=flare, batch_size=2, shuffle=False, num_workers=0, collate_fn=my_collate)
    for train_iter, pack in enumerate(train_loader):
        img_ = pack['image']
        label_ = pack['label']
        name_ = pack['name']
        print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))