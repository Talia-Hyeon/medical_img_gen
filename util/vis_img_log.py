import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')


def make_tot_log_npy(log_root):
    os.makedirs(f'{log_root}/mean', exist_ok=True)
    tot_l1 = list()
    tot_l2 = list()
    tot_bn = list()
    tot_dice = list()

    for path, dirs, filenames in os.walk(log_root):
        for filename in filenames:
            file = osp.join(path, filename)
            (basename, extension) = osp.splitext(filename)
            if basename == 'l1':
                l1_npy = np.load(file)
                tot_l1.append(l1_npy)

            elif basename == 'l2':
                l2_npy = np.load(file)
                tot_l2.append(l2_npy)

            elif basename == 'bn':
                bn_npy = np.load(file)
                tot_bn.append(bn_npy)

            elif basename == 'dice':
                dice_npy = np.load(file)
                tot_dice.append(dice_npy)

    l1_log = np.array(tot_l1)
    l1_mean = l1_log.mean(axis=0)
    np.save(f'{log_root}/mean/l1.npy', l1_mean)

    l2_log = np.array(tot_l2)
    l2_mean = l2_log.mean(axis=0)
    np.save(f'{log_root}/mean/l2.npy', l2_mean)

    bn_log = np.array(tot_bn)
    bn_mean = bn_log.mean(axis=0)
    np.save(f'{log_root}/mean/bn.npy', bn_mean)

    dice_log = np.array(tot_dice)
    dice_mean = dice_log.mean(axis=0)
    np.save(f'{log_root}/mean/dice.npy', dice_mean)


def make_plot(root):
    l1_log = np.load(f'{root}/l1.npy')
    l2_log = np.load(f'{root}/l2.npy')
    bn_log = np.load(f'{root}/bn.npy')
    dice_log = np.load(f'{root}/dice.npy')

    plt.plot(l1_log, 'y', label='L1')
    plt.plot(l2_log, 'g', label='L2')
    plt.plot(bn_log, 'b', label='Batch_Norm')
    plt.plot(dice_log, 'r', label='Dice')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{root}/Log of generating images.png')


if __name__ == '__main__':
    log_root = '../log_img/di_mask dice 1/1'
    make_tot_log_npy(log_root)
    make_plot(f'{log_root}/mean')