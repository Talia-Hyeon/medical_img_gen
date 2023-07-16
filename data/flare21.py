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
    def __init__(self, root, split='train', task_id=1, crop_size=(96, 96, 96), cur_epoch=0, batch_size=8, milestones=[40, 80, 120]):
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

        iter_per_epoch = (len(self.files)+batch_size-1)//batch_size
        self.cur_iter = iter_per_epoch*cur_epoch
        self.milestones = [iter_per_epoch*e for e in milestones]
 
        print("{}'s {} images are loaded!".format(self.split, len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, name = self.files[index]

        # augmentation
        if self.split == 'train':
            
            prob = self.get_centor_crop_prob()

            if np.random.uniform() < prob:
                image, label = center_crop_3d(image, label, self.crop_h, self.crop_w, self.crop_d)
            else:
                iamge, label = random_crop_3d(image, label, self.crop_h, self.crop_w, self.crop_d)

            # 50% flip
            if np.random.rand(1) <= 0.5:  # W
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            if np.random.rand(1) <= 0.5:  # H
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            if np.random.rand(1) <= 0.5:  # D
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]

        if self.split == 'train':
            label = label_to_binary(label, self.task_id)
        else:
            # extend label's channel for val/test
            label = extend_channel_classes(label, self.task_id)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        self.cur_iter += 1

        return image, label, name
    
    def get_centor_crop_prob(self):
        if self.cur_iter < self.milestones[0]:
            return 1.0
        elif self.cur_iter < self.milestones[1]:
            return 0.8
        elif self.cur_iter < self.milestones[2]:
            return 0.6
        else:
            return 0.4

    def load_data(self, data_l, image_path, label_path):
        all_files = []
        for i, item in enumerate(data_l):
            img_file = osp.join(image_path, item)
            label_item = item.replace('_0000', '')
            label_file = osp.join(label_path, label_item)

            # read nii file
            imageNII = nib.load(img_file)
            labelNII = nib.load(label_file)
            image = imageNII.get_fdata()
            label = labelNII.get_fdata()
            name = item

            # normalization
            image = truncate(image)  # -1 <= image <= 1

            # add channel
            image = image[np.newaxis, :]
            label = label[np.newaxis, :]

            # Channel x Depth x H x W
            image = image.transpose((0, 3, 1, 2))
            label = label.transpose((0, 3, 1, 2))

            if self.split == 'train':
                all_files.append((image, label, name))
            else:
                _, d, h, w = image.shape
                num_d = (d+self.crop_d-1)//self.crop_d
                num_h = (h+self.crop_h-1)//self.crop_h
                num_w = (w+self.crop_w-1)//self.crop_w
                
                # crop
                for k in range(num_d):
                    for j in range(num_w):
                        for i in range(num_h):
                            pos_h = self.crop_h*i
                            pos_w = self.crop_w*j 
                            pos_d = self.crop_d*k

                            croped_img = image[:, pos_d:pos_d+self.crop_d, pos_h:pos_h+self.crop_h, pos_w:pos_w+self.crop_w]
                            croped_lbl = label[:, pos_d:pos_d+self.crop_d, pos_h:pos_h+self.crop_h, pos_w:pos_w+self.crop_w]

                            croped_img = pad_image(croped_img, [self.crop_d, self.crop_h, self.crop_w])
                            croped_lbl = pad_image(croped_lbl, [self.crop_d, self.crop_h, self.crop_w])

                            all_files.append((croped_img, croped_lbl, name))
        
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

def center_crop_3d(image, label, crop_h, crop_w, crop_d):
    _, depth, height, width = image.shape
    h0 = (height - crop_h) // 2
    h1 = h0 + crop_h
    w0 = (width - crop_w) // 2
    w1 = w0 + crop_w
    d0 = (depth - crop_d) // 2
    d1 = d0 + crop_d
    image = image[:, d0: d1, h0: h1, w0: w1]
    label = label[:, d0: d1, h0: h1, w0: w1]
    return image, label

def random_crop_3d(image, label, crop_h, crop_w, crop_d):
    _, depth, height, width = image.shape
    h0 = np.random.randint(0, height-crop_h)
    h1 = h0 + crop_h
    w0 = np.random.randint(0, width-crop_w)
    w1 = w0 + crop_w
    d0 = np.random.randint(0, depth-crop_d)
    d1 = d0 + crop_d
    image = image[:, d0: d1, h0: h1, w0: w1]
    label = label[:, d0: d1, h0: h1, w0: w1]
    return image, label

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    _, d, h, w = img.shape
    dept_missing = math.ceil(target_size[0] - d)
    rows_missing = math.ceil(target_size[1] - h)
    cols_missing = math.ceil(target_size[2] - w)
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0
    if dept_missing < 0:
        dept_missing = 0

    padded_img = np.pad(img, ((0, 0), (0, dept_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def label_to_binary(label, task_id):
    label_list = []

    label_backg = label.copy()
    label_backg[np.where(label != 0)] = 0
    label_backg[np.where(label == 0)] = 1
    label_list.append(label_backg)

    label_foreg = label.copy()
    label_foreg[np.where(label != task_id)] = 0
    label_foreg[np.where(label == task_id)] = 1
    label_list.append(label_foreg)

    stacked_label = np.stack(label_list, axis=1)
    stacked_label = np.squeeze(stacked_label)
    return stacked_label


def extend_channel_classes(label, num_classes):
    label_list = []
    for i in range(0, num_classes + 1):
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
    # print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))

    # flare = FLAREDataSet(root='../dataset/FLARE21', split='train', task_id=4)
    # train_loader = data.DataLoader(dataset=flare, batch_size=1, shuffle=False, num_workers=4, collate_fn=my_collate)
    # for train_iter, pack in enumerate(train_loader):
    #     img_ = pack['image']
    #     label_ = pack['label']
    #     name_ = pack['name']
    #     print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))