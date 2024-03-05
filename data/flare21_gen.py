import os
import os.path as osp
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data

sys.path.append('..')
from data.flare21 import FLAREDataSet
from util import decode_segmap, find_best_view


class FLARE_Mask(FLAREDataSet):
    def __init__(self, root='../dataset/FLARE_Dataset/train'):
        self.root = root

        # read path
        self.files = self.load_data()
        print("{} images are loaded!".format(len(self.files)))

    def __getitem__(self, index):
        label = self.files[index]['label']
        name = self.files[index]['name']
        # load numpy array
        label = np.load(label)

        # 50% flip
        label = self.random_filp(label)

        # extend label's channel for val/test
        label = self.extend_channel_classes(label)
        label = label.astype(np.float32)
        return label, name

    def load_data(self):
        all_files = []
        mask_path = osp.join(self.root, 'mask')
        mask_list = os.listdir(mask_path)

        for i, item in enumerate(mask_list):
            mask_file = osp.join(mask_path, item)
            name = item.split('.')[0]
            all_files.append({"label": mask_file, "name": name})

        return all_files

    def random_filp(self, mask):
        if np.random.rand(1) <= 0.5:  # W
            mask = mask[:, :, :, ::-1]
        if np.random.rand(1) <= 0.5:  # H
            mask = mask[:, :, ::-1, :]
        if np.random.rand(1) <= 0.5:  # D
            mask = mask[:, ::-1, :, :]
        return mask


def visualization(mask, name, root, num_classes):
    # delete batch & channel
    mask = torch.squeeze(mask).numpy()
    mask = np.squeeze(mask)
    mask = np.argmax(mask, axis=0)

    # slice into the best view
    max_score_idx = find_best_view(mask, num_classes)
    mask = mask[max_score_idx, :, :]
    col_mask = decode_segmap(mask, num_classes)

    plt.figure()
    plt.imshow(col_mask)
    plt.title(name)
    plt.savefig(f'{root}/{name}.png')
    plt.close()


if __name__ == '__main__':
    task_id = 1
    save_path = '../fig/preprocessed_data'
    os.makedirs(save_path, exist_ok=True)
    mask_set = FLARE_Mask(task_id=task_id)
    mask_loader = data.DataLoader(dataset=mask_set, batch_size=1, shuffle=False)
    for mask_iter, pack in enumerate(mask_loader):
        mask_ = pack[0]
        name_ = pack[1][0]
        visualization(mask_, name_, save_path, 5)
