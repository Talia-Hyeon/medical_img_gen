import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data

sys.path.append('..')
from util import visualization
from data.flare21 import random_flip, load_data, FLAREDataSet


class FAKEDataSet(FLAREDataSet):
    def __init__(self, root):
        self.root = root

        # load data
        image_path = osp.join(self.root, 'img')
        img_list = os.listdir(image_path)
        self.files = load_data(img_list, image_path)
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = datafiles['image']
        label = datafiles['label']
        name = datafiles['name']

        # load numpy array
        image = np.load(image)
        label = np.load(label)

        # 50% flip
        image, label = random_flip(image, label)

        # extend label's channel for val/test
        label = self.extend_channel_classes(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name


if __name__ == '__main__':
    # train_type = 'hrhf'
    train_type = 'di_mask'
    save_path = f'../fig/gen_img/{train_type}/'
    os.makedirs(save_path, exist_ok=True)
    val_path = f'../sample/{train_type}'
    val_data = FAKEDataSet(root=val_path)
    val_loader = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)
    for val_iter, pack in enumerate(val_loader):
        img_ = pack[0]
        label_ = pack[1]
        name = pack[2][0]
        visualization(img_, label_, save_path, name, num_classes=5)
