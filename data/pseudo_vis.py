import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data

sys.path.append('..')
from util import find_best_view


class PseudoVis(data.Dataset):
    def __init__(self, npy_root, png_root):
        self.npy_root = npy_root
        self.png_root = png_root

        self.npy_list = self.make_npy_list(rootdir=self.npy_root, suffix='npy')

    def __len__(self):
        return len(self.files)

    def make_npy_list(self, rootdir=".", suffix=""):
        npy_list = list()

        for path, dirs, filenames in os.walk(rootdir):
            for filename in filenames:
                if filename.endswith(suffix):
                    file = osp.join(path, filename)
                    if file.split(os.sep)[-2] == 'img':
                        npy_list.append(file)

        return npy_list

    def save_png(self, npy_img_path):
        npy_label_path = npy_img_path.replace('img', 'mask')

        img_path_split = npy_img_path.split(os.sep)
        img_iter = img_path_split[-1]
        img_name = img_path_split[-3]
        img_dir = osp.join(self.png_root, img_name)
        os.makedirs(img_dir, exist_ok=True)

        # load numpy array
        image = np.load(npy_img_path)
        label = np.load(npy_label_path)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        self.visualization(image, label, img_iter, img_dir)

    def visualization(self, npy_img, npy_label, img_iter, img_dir):


if __name__ == '__main__':
    train_type = 'di_mask'
    save_path = f'../fig/gen_img/{train_type}/'
    os.makedirs(save_path, exist_ok=True)
    npy_path = f'../sample/{train_type}'
    os.makedirs(npy_path, exist_ok=True)

    pseudo_images = PseudoVis(npy_root=npy_path, png_root=save_path)

    for npy_path in pseudo_images.npy_list:
        pseudo_images.save_png(npy_path)
