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
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.ylim(0, 10)
    plt.savefig(f'{root}/Log of generating images.png')


def save_tot_loss(root):
    l1_log = np.load(f'{root}/l1.npy')
    l2_log = np.load(f'{root}/l2.npy')
    bn_log = np.load(f'{root}/bn.npy')
    dice_log = np.load(f'{root}/dice.npy')
    tot_log = l1_log + l2_log + bn_log + dice_log * 10.0
    np.save(f'{root}/tot.npy', tot_log)


def dice_plot(root):
    dice_list = os.listdir(root)
    for i in range(len(dice_list)):
        path = osp.join(root, dice_list[i])

        if dice_list[i].find('1e-2') != -1:
            dice_em2 = np.load(path)
        elif dice_list[i].find('10') != -1:
            dice_10 = np.load(path)
        else:
            dice_1 = np.load(path)

    plt.plot(dice_em2, 'g', label='Weight of Dice Loss: 1e-2')
    plt.plot(dice_1, 'b', label='Weight of Dice Loss: 1.0')
    plt.plot(dice_10, 'r', label='Weight of Dice Loss: 10.0')
    plt.legend(loc='upper right')
    # plt.yscale('log')
    plt.ylim(0, 0.6)
    plt.grid(True)
    plt.savefig(f'{root}/Log of bn.png')


if __name__ == '__main__':
    # log_root = '../log_img/di_mask dice 1'
    # make_tot_log_npy(log_root)
    # make_plot(f'{log_root}/mean')

    # save_tot_loss('../log_img/di_mask dice 1e-2')
    root = '../log_img/bn'
    dice_plot(root)
