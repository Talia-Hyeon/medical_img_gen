import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data


class PseudoVis(data.Dataset):
    def __init__(self, npy_root, png_root):
        self.npy_root = npy_root
        self.png_root = png_root

        npy_list = self.make_npy_list(rootdir=self.root, suffix='npy')  # 라벨 제


    def __len__(self):
        return len(self.files)

    def make_npy_list(self, rootdir=".", suffix=""):
        return [
            os.path.join(path, filename)
            for path, dirs, filenames in os.walk(rootdir)
            # rootdir: ../sample/di_mask/, img_name: train_002_0, [imgname.png,...]
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def make_png_list(self, img):
        img_path = img.split(os.sep)
        img_path[-1] = img_path[-1].replace('npy', 'png')
        save_img_path = osp.join(self.png_root, img_path[-3], img_path[-1])

