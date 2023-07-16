import os
import os.path as osp
import random
import math

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data
import nibabel as nib
from skimage.transform import resize
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose

global index_organs
index_organs = ['background', 'liver', 'kidney', 'spleen', 'pancreas']


class FLAREDataSet(data.Dataset):
    def __init__(self, root, split='train', task_id=1, crop_size=(96, 96, 96)):
        self.root = root
        self.split = split
        self.task_id = task_id
        self.crop_d, self.crop_h, self.crop_w = crop_size

        print("Start preprocessing....")
        # read path
        image_path = osp.join(self.root, 'TrainingImg')
        label_path = osp.join(self.root, 'TrainingMask')
        img_list = os.listdir(image_path)

        # split train/val/test set
        train_data, rest_data = train_test_split(img_list, test_size=0.20, shuffle=True, random_state=0)
        val_data, test_data = train_test_split(rest_data, test_size=0.50, shuffle=True, random_state=0)

        if self.split == 'train':
            all_files = self.load_data(train_data, image_path, label_path)
            self.files = all_files

        elif self.split == 'val':
            all_files = self.load_data(val_data, image_path, label_path)
            self.files = all_files

        elif self.split == 'test':
            all_files = self.load_data(test_data, image_path, label_path)
            self.files = all_files

        print("{}'s {} images are loaded!".format(self.split, len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])
        image = imageNII.get_fdata()
        label = labelNII.get_fdata()
        name = datafiles["name"]

        # scale
        if self.split == 'train' and np.random.uniform() < 0.2:
            scaler = np.random.uniform(0.7, 1.4)
        else:
            scaler = 1

        # pad
        image = pad_image(image, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])
        label = pad_image(label, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])

        image, label = center_crop_3d(image, label, self.crop_h, self.crop_w, self.crop_d)

        # normalization
        image = truncate(image)  # -1 <= image <= 1

        # add channel
        image = image[np.newaxis, :]
        label = label[np.newaxis, :]

        # Channel x Depth x H x W
        image = image.transpose((0, 3, 1, 2))
        label = label.transpose((0, 3, 1, 2))

        # 50% flip
        if self.split == 'train':
            if np.random.rand(1) <= 0.5:  # W
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            if np.random.rand(1) <= 0.5:  # H
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            if np.random.rand(1) <= 0.5:  # D
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]

        # adjust shape
        d, h, w = image.shape[-3:]
        if scaler != 1 or d != self.crop_d or h != self.crop_h or w != self.crop_w:
            image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (1, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0,
                           clip=True, preserve_range=True)

        # extend label's channel for val/test
        label = extend_channel_classes(label, self.task_id)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name

    def load_data(self, data_l, image_path, label_path):
        all_files = []
        for i, item in enumerate(data_l):
            img_file = osp.join(image_path, item)
            label_item = item.replace('_0000', '')
            label_file = osp.join(label_path, label_item)

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item
            })
        return all_files


def truncate(CT):
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


def locate_bbx(crop_d, crop_h, crop_w, label, scaler, bbx):
    scale_d = int(crop_d * scaler)
    scale_h = int(crop_h * scaler)
    scale_w = int(crop_w * scaler)

    img_h, img_w, img_d = label.shape
    boud_h, boud_w, boud_d = bbx

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

    if (bbx_w_max - bbx_w_min) <= scale_w:
        bbx_w_maxt = bbx_w_max + math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
        bbx_w_mint = bbx_w_min - math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
        if bbx_w_mint < 0:
            bbx_w_maxt -= bbx_w_mint

    if (bbx_d_max - bbx_d_min) <= scale_d:
        bbx_d_maxt = bbx_d_max + math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
        bbx_d_mint = bbx_d_min - math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
        if bbx_d_mint < 0:
            bbx_d_maxt -= bbx_d_mint

    # no patch outside the image
    d0 = random.randint(0, img_d - scale_d)
    h0 = random.randint(0, img_h - scale_h)
    w0 = random.randint(0, img_w - scale_w)

    d1 = d0 + scale_d
    h1 = h0 + scale_h
    w1 = w0 + scale_w
    return [h0, h1, w0, w1, d0, d1]


def center_crop_3d(image, label, crop_h, crop_w, crop_d):
    height, width, depth = image.shape
    h0 = (height - crop_h) // 2
    h1 = h0 + crop_h
    w0 = (width - crop_w) // 2
    w1 = w0 + crop_w
    d0 = (depth - crop_d) // 2
    d1 = d0 + crop_d
    image = image[h0: h1, w0: w1, d0: d1]
    label = label[h0: h1, w0: w1, d0: d1]
    return image, label


def pad_image(img, target_size):
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


def extend_channel_classes(label, num_classes):
    label_list = []
    for i in range(1, num_classes + 1):
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
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True,
                              p_per_channel=0.5, p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15,
                                             p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True,
                                        retain_stats=True, p_per_sample=0.15, data_key="image"))

    # compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'image': image,
                 'label': label,
                 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


if __name__ == '__main__':
    flare = FLAREDataSet(root='../dataset/FLARE21', split='train', task_id=4)
    img_, label_, name_ = flare[0]
