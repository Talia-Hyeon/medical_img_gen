import os
import os.path as osp
import random
import sys

import numpy as np
from torch.utils import data

sys.path.append('..')
from util import visualization

global index_organs
index_organs = ['background', 'liver', 'kidney', 'spleen', 'pancreas']


class FLAREDataSet(data.Dataset):
    def __init__(self, root='../../MOSInversion/dataset/FLARE_Dataset', split='train', task_id=4):
        self.root = root
        self.split = split
        self.task_id = task_id

        # read image path
        if self.split == 'train':
            image_path = osp.join(self.root, self.split, str(self.task_id), 'img')
        else:
            image_path = osp.join(self.root, self.split, 'img')

        img_list = os.listdir(image_path)
        self.files = load_data(img_list, image_path)
        print("{}'s {} images are loaded!".format(self.split, len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = self.files[index]['image']
        label = self.files[index]['label']
        name = self.files[index]['name']

        # load numpy array
        image = np.load(image)
        label = np.load(label)

        if self.split == 'train':
            # 50% flip
            image, label = random_flip(image, label)

        # extend label's channel
        label = self.extend_channel_classes(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name

    def extend_channel_classes(self, label):
        label_list = []
        bg = np.ones_like(label)
        for i in range(1, self.task_id + 1):  # 1: from foreground
            label_i = label.copy()
            label_i[label == i] = 1
            label_i[label != i] = 0
            bg -= label_i
            label_list.append(label_i)
        # bg = np.clip(bg, 0, 1)
        label_list = [bg] + label_list
        stacked_label = np.concatenate(label_list, axis=0)
        return stacked_label


def random_flip(image, label):
    if np.random.rand(1) <= 0.5:  # D
        image = image[:, :, :, ::-1]
        label = label[:, :, :, ::-1]
    if np.random.rand(1) <= 0.5:  # H
        image = image[:, :, ::-1, :]
        label = label[:, :, ::-1, :]
    if np.random.rand(1) <= 0.5:  # W
        image = image[:, ::-1, :, :]
        label = label[:, ::-1, :, :]
    return image, label


def load_data(data_l, image_path):
    all_files = []
    for i, item in enumerate(data_l):
        img_file = osp.join(image_path, item)
        label_file = img_file.replace('img', 'mask')
        name = item.split('.')[0]

        data = {"image": img_file,
                "label": label_file,
                "name": name}
        all_files.append(data)
    return all_files


if __name__ == '__main__':
    for task_id in range(1, 5):
        flare = FLAREDataSet(split='train', task_id=task_id)

    # task_id = 4
    # save_path = '../fig/preprocessed_data/flare'
    # os.makedirs(save_path, exist_ok=True)
    # flare = FLAREDataSet(root='../dataset/FLARE_Dataset_10', split='val', task_id=task_id)
    # val_loader = data.DataLoader(dataset=flare, batch_size=1, shuffle=False)
    # for val_iter, pack in enumerate(val_loader):
    #     img_ = pack[0]
    #     label_ = pack[1]
    #     name_ = pack[2][0]
    #     visualization(img_, label_, save_path, name_, 5)
