import sys
import os
import os.path as osp

import torch
from torch.utils import data
import numpy as np

sys.path.append('..')
from data.flare21 import random_flip, load_data
from util import visualization


class BinaryDataSet(data.Dataset):
    def __init__(self, root='../../MOSInversion/dataset/OneOrgan_Dataset', task_id=1):
        self.task_id = task_id

        # read image path
        image_path = osp.join(root, str(self.task_id), 'img')
        img_list = os.listdir(image_path)

        # load data
        self.files = load_data(img_list, image_path)
        print("{} images are loaded!".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = self.files[index]['image']
        label = self.files[index]['label']
        name = self.files[index]['name']

        # load numpy array
        image = np.load(image)
        label = np.load(label)

        # 50% flip
        image, label = random_flip(image, label)

        # indexing label
        label = self.add_background(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name

    def add_background(self, label):
        bg = np.ones_like(label)
        bg -= label
        stacked_label = np.concatenate([bg, label], axis=0)
        return stacked_label


if __name__ == '__main__':
    for task_id in range(1, 5):
        flare = BinaryDataSet(task_id=task_id)

    # task_id = 1
    # save_path = f'../fig/preprocessed_data/one_organ/{str(task_id)}'
    # os.makedirs(save_path, exist_ok=True)
    # one_organ = BinaryDataSet(root='../dataset/OneOrgan_Dataset', task_id=task_id)
    # train_loader = data.DataLoader(dataset=one_organ, batch_size=1)
    # for train_iter, pack in enumerate(train_loader):
    #     img_ = pack[0]
    #     label_ = pack[1]
    #     name_ = pack[2][0]
    #     visualization(img_, label_, save_path, name_, 2)