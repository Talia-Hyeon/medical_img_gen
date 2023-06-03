import os
import os.path as osp
import random
import math
import sys

import numpy as np
from torch.utils import data
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
}

valid_dataset = {
    'liver': 0,
    'spleen': 1,
    'pancreas': 2,
    'kidney': 3
}  # background class 없앰??


class BTCVDataSet(data.Dataset):
    def __init__(self, root, task_name='liver', crop_size=(64, 256, 256), mean=(128, 128, 128), scale=True,
                 mirror=False, ignore_label=255, Train=True):
        self.root = root
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.Train = Train
        self.task_name = task_name
        self.task_id = valid_dataset[self.task_name]

        spacing = {
            0: [0.8, 0.8, 1.5],
            1: [0.8, 0.8, 1.5],
            2: [0.8, 0.8, 1.5],
            3: [0.8, 0.8, 1.5]
        }

        # spacing = {0: [3.0, 1.5, 1.5]} for test
        # crop_size=(64, 256, 256) for test
        # crop_size와 spacing 관계?

        print("Start preprocessing....")
        # load data
        image_path = os.path.join(self.root, 'img')
        label_path = os.path.join(self.root, 'label')

        img_list = os.listdir(os.path.join(self.root, 'img'))
        all_files = []
        for i, item in enumerate(img_list):
            img_file = osp.join(image_path, item)
            label_item = item.replace('img', 'label')
            label_file = osp.join(label_path, label_item)

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item,
                "task_id": self.task_id,
                "spacing": spacing[self.task_id],
            })

        # split train-test set
        train_X, val_X = train_test_split(all_files, test_size=0.20, shuffle=True, random_state=0)
        if self.Train == True:
            self.files = train_X
        else:
            self.files = val_X
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):  # 목적?
        # min_HU = -200
        # max_HU = 250
        # subtract = 37.5
        # divide = 250-37.5
        # in btcv, they use
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

    def locate_bbx(self, label, scaler, bbx):  # for patch 단위?
        scale_d = int(self.crop_d * scaler)  # crop 기준 위치
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

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, np.max([bbx_d_max - scale_d, bbx_d_min]))
            h0 = random.randint(bbx_h_min, np.max([bbx_h_max - scale_h, bbx_h_min]))
            w0 = random.randint(bbx_w_min, np.max([bbx_w_max - scale_w, bbx_w_min]))
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

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

        # boud_h, boud_w, boud_d = np.where(label >= 1)  # background 아닌
        # self.files[index].setdefault("bbx", [boud_h, boud_w, boud_d])

        name = datafiles["name"]
        task_id = datafiles["task_id"]
        organ_name = [k for k, v in valid_dataset.items() if v == task_id][0]

        # new class of organ: 1(학습 중) , not: 0
        # merge right-left kidney into same label
        if organ_name != 'kidney':
            label_id = BTCV_label_num[organ_name]
            # np.set_printoptions(threshold=sys.maxsize)
            label[np.where(label != label_id)] = 0

        else:
            label_id1 = BTCV_label_num['right kidney']  # 2
            label_id2 = BTCV_label_num['left kidney']  # 3
            label[np.where((label != label_id1) & (label != label_id2))] = 0
        label[np.where(label != 0)] = 1

        image = self.pad_image(image, [self.crop_h, self.crop_w, self.crop_d])
        label = self.pad_image(label, [self.crop_h, self.crop_w, self.crop_d])

        image = self.truncate(image)
        label = self.id2trainId(label, task_id)  # original has tumor label

        image = image[np.newaxis, :]

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, task_id, labelNII.affine

    def id2trainId(self, label, task_id):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)
        results_map[0, :, :, :] = label
        results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        return results_map


def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5, order_downsample=0,
                                                        order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False,
                                        per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):  # dataset이 variable length(shape)이면 collate_fn을 꼭 사용
    image, label, name, task_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)
    data_dict = {'image': image, 'label': label, 'name': name, 'task_id': task_id}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


if __name__ == '__main__':
    btcv = BTCVDataSet(root='../dataset/BTCV/Trainset')
    sample = btcv[0]