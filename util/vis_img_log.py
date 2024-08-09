import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('..')


def make_tot_log_npy(root_dir, weight):
    log_root = osp.join(root_dir, 'di_mask dice ' + weight)
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
    np.save(f'{root_dir}/l1/{weight}.npy', l1_mean)

    l2_log = np.array(tot_l2)
    l2_mean = l2_log.mean(axis=0)
    np.save(f'{root_dir}/l2/{weight}.npy', l2_mean)

    bn_log = np.array(tot_bn)
    bn_mean = bn_log.mean(axis=0)
    np.save(f'{root_dir}/bn/{weight}.npy', bn_mean)

    dice_log = np.array(tot_dice)
    dice_mean = dice_log.mean(axis=0)
    if weight == '1e-2':
        dice_mean = dice_mean * 100.0
    np.save(f'{root_dir}/dice/{weight}.npy', dice_mean)


def save_tot_loss(root, weight):
    l1_log = np.load(f'{root}/l1/{weight}.npy')
    l2_log = np.load(f'{root}/l2/{weight}.npy')
    bn_log = np.load(f'{root}/bn/{weight}.npy')
    dice_log = np.load(f'{root}/dice/{weight}.npy')
    tot_log = l1_log + l2_log + bn_log + dice_log
    np.save(f'{root}/tot/{weight}.npy', tot_log)


def loss_plot(root, loss_type):
    loss_path = osp.join(root, loss_type)
    weight_list = os.listdir(loss_path)
    for i in weight_list:
        path = osp.join(loss_path, i)
        if i.find('1e-2') != -1:
            weight_em2 = np.load(path)
        elif i.find('10') != -1:
            weight_10 = np.load(path)
            if loss_type == 'dice':
                weight_10 = weight_10 / 10.0
        else:
            weight_1 = np.load(path)

    plt.plot(weight_em2, 'g', label='1e-2')
    plt.plot(weight_1, 'b', label='1.0')
    plt.plot(weight_10, 'r', label='10.0')
    plt.legend(loc='upper right')
    plt.yscale('log')
    # plt.ylim(0, 12)
    plt.grid(True)
    plt.savefig(f'{root}/{loss_type} Loss.png')
    plt.close()


def numpy2csv(root, loss_type):
    loss_path = osp.join(root, loss_type)
    weight3_list = os.listdir(loss_path)
    # load loss and save in list
    loss_list = [np.load(osp.join(loss_path, file)) for file in weight3_list]

    # check shape
    if not all(loss.shape == loss_list[0].shape for loss in loss_list):
        raise ValueError("all weights must have same shape.")

    # add first row
    data = {os.path.splitext(file)[0]: loss.flatten() for file, loss in zip(weight3_list, loss_list)}

    df = pd.DataFrame(data)
    df.to_csv(f'{root}/{loss_type}.csv', index=False)


if __name__ == '__main__':
    root_list = ['1e-2', '1', '10']
    root_dir = '../log_img'

    # os.makedirs(f'{root_dir}/l1', exist_ok=True)
    # os.makedirs(f'{root_dir}/l2', exist_ok=True)
    # os.makedirs(f'{root_dir}/bn', exist_ok=True)
    # os.makedirs(f'{root_dir}/dice', exist_ok=True)
    # os.makedirs(f'{root_dir}/tot', exist_ok=True)
    #
    # for weight in root_list:
    #     make_tot_log_npy(root_dir, weight)
    #     save_tot_loss(root_dir, weight)

    loss_list = ['l1', 'l2', 'bn', 'dice', 'tot']
    for loss_type in loss_list:
        # loss_plot(root_dir, loss_type)
        numpy2csv(root_dir, loss_type)
