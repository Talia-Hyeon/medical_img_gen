import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data

sys.path.append('..')
from util.util import find_best_view
from util.util import decode_segmap


class PseudoVis(data.Dataset):
    def __init__(self, npy_root, png_root, num_classes):
        self.npy_root = npy_root
        self.png_root = png_root
        self.num_classes = num_classes

        self.npy_list = self.make_npy_list()

    def __len__(self):
        return len(self.files)

    def make_npy_list(self):
        npy_list = list()

        for path, dirs, filenames in os.walk(self.npy_root):
            for filename in filenames:
                if filename.endswith('npy'):
                    file = osp.join(path, filename)
                    if file.split(os.sep)[-2] == 'img':
                        npy_list.append(file)

        return npy_list

    def save_png(self, npy_img_path):
        npy_label_path = npy_img_path.replace('img', 'mask')
        npy_target_path = npy_img_path.replace('img', 'label')

        img_path_split = npy_img_path.split(os.sep)
        img_iter = img_path_split[-1]
        img_name = img_path_split[-3]
        img_dir = osp.join(self.png_root, img_name)
        os.makedirs(img_dir, exist_ok=True)

        # load numpy array
        image = np.load(npy_img_path)
        label = np.load(npy_label_path)
        target = np.load(npy_target_path)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        target = target.astype(np.float32)

        self.visualization(image, label, target, img_iter, img_dir)

    def visualization(self, npy_img, npy_label, npy_target, img_iter, img_dir):
        # remove batch
        npy_img = np.squeeze(npy_img)
        npy_label = np.squeeze(npy_label)
        npy_target = np.squeeze(npy_target)

        # slice into the best view
        max_score_idx = find_best_view(npy_target, self.num_classes)
        img = npy_img[max_score_idx, :, :]
        label = npy_label[max_score_idx, :, :]
        target = npy_target[max_score_idx, :, :]

        col_label = decode_segmap(label, self.num_classes)
        col_target = decode_segmap(target, self.num_classes)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Image')
        plt.subplot(1, 3, 2)
        plt.imshow(col_label)
        plt.title('Pseudo GT')
        plt.subplot(1, 3, 3)
        plt.imshow(col_target)
        plt.title('Real GT')
        plt.savefig(f'{img_dir}/{img_iter}.png')
        plt.close()


if __name__ == '__main__':
    train_type = 'hrhf'
    save_path = f'../fig/gen_img/{train_type}/'
    os.makedirs(save_path, exist_ok=True)
    npy_path = f'../sample/{train_type}'
    os.makedirs(npy_path, exist_ok=True)

    pseudo_images = PseudoVis(npy_root=npy_path, png_root=save_path, num_classes=5)

    # pseudo_images.save_png(pseudo_images.npy_list[0])
    for npy_path in pseudo_images.npy_list:
        pseudo_images.save_png(npy_path)
