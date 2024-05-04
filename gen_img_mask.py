import argparse
import random
import os
import timeit
from itertools import cycle
from copy import deepcopy
import multiprocessing as mp

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from model.deepInversion_3d import DeepInversionFeatureHook, get_image_prior_losses, save_nyp, save_nyp_mask
from data.flare21_gen import FLARE_Mask
from loss_functions.score import DiceLoss
from util.util import load_model
from gen_img_class_vector import image_size

global gen_loss_weight
gen_loss_weight = {'dice_loss': 1.0, 'loss_bn': 1.0, 'loss_var_l1': 2.5e-5, 'loss_var_l2': 3e-8}


def gen_img_mask(args, num, gen_l1_loss, gen_l2_loss, gen_bn_loss, gen_dice_loss, lock):
    # hyper-parameter
    train_type = args.train_type
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_workers = args.num_workers
    n_iters = args.gen_epochs
    cnt = args.cnt
    num.value = cnt
    n_imgs = args.num_imgs
    dice_weight = args.dice

    device_ids = [i for i in range(torch.cuda.device_count())]
    pixels = image_size[0] * image_size[1] * image_size[2]

    cudnn.benchmark = True

    # make directory
    root_p = f"./sample/{train_type}/"
    # logger
    log_loss = f'./fig/{train_type}/'
    os.makedirs(log_loss, exist_ok=True)

    # load the pretrained model
    pretrained = load_model()
    pretrained.eval()
    pretraineds = [deepcopy(pretrained).to(device_ids[i]) for i in range(len(device_ids))]

    # dice loss
    loss_fns = [DiceLoss(num_classes=num_classes).to(device_ids[i]) for i in range(len(device_ids))]

    # define the dataset
    real_data = FLARE_Mask(root='./dataset/FLARE_Dataset/train')
    real_dataloader = data.DataLoader(dataset=real_data, batch_size=batch_size, drop_last=True,
                                      num_workers=num_workers, shuffle=True)
    real_dataloader_infinite_iter = iter(cycle(real_dataloader))

    # generate fake images
    while cnt < n_imgs:
        start = timeit.default_timer()
        fake_xs = [torch.randn([1, 1] + list(image_size), requires_grad=True, device=device_ids[i]) for i in
                   range(len(device_ids))]
        optimizers = [torch.optim.Adam([fake_xs[i]], lr=0.1) for i in range(len(device_ids))]

        # define the mask
        mask, name = next(real_dataloader_infinite_iter)

        children = list()

        for pid in range(len(device_ids)):
            child = mp.Process(target=gen_img, args=(
                pid, pretraineds[pid], fake_xs[pid], optimizers[pid], mask[pid].to(device_ids[pid]), name[pid],
                loss_fns[pid], root_p, pixels, n_iters, n_imgs, num, gen_l1_loss, gen_l2_loss, gen_bn_loss,
                gen_dice_loss, lock, dice_weight))
            children.append(child)
            child.start()

        for child in children:
            child.join()

        cnt = num.value

        end = timeit.default_timer()
        print(f'generataion end: {(end - start):.3f} s')

    # save loss of generation in figure
    l1_log = np.array(gen_l1_loss)
    l1_log = l1_log.mean(axis=0)
    l2_log = np.array(gen_l2_loss)
    l2_log = l2_log.mean(axis=0)
    bn_log = np.array(gen_bn_loss)
    bn_log = bn_log.mean(axis=0)
    dice_log = np.array(gen_dice_loss)
    dice_log = dice_log.mean(axis=0)

    plt.plot(l1_log, 'y', label='L1')
    plt.plot(l2_log, 'g', label='L2')
    plt.plot(bn_log, 'b', label='Batch_Norm')
    plt.plot(dice_log, 'r', label='Dice')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{log_loss}/Log of generating images.png')


def gen_img(pid, pretrained, fake_x, optimizer, mask, name, loss_fn,
            root_p, pixels, n_iters, n_imgs, num, gen_l1_loss, gen_l2_loss, gen_bn_loss,
            gen_dice_loss, lock, dice_weight):
    # make dirs
    root_p = root_p + '/' + name
    os.makedirs(f'{root_p}/img', exist_ok=True)
    os.makedirs(f'{root_p}/mask', exist_ok=True)
    os.makedirs(f'{root_p}/label', exist_ok=True)

    # log
    gen_loss_log = {'L1': [], 'L2': [], 'Batch_Norm': [], 'Dice': []}

    # add batch
    mask = torch.unsqueeze(mask, dim=0)

    # hook pretrained model
    loss_r_feature_layers = []
    for module in pretrained.modules():
        if isinstance(module, nn.BatchNorm3d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # rescale values for hook
    rescale = [10.] + [1. for _ in range(len(loss_r_feature_layers) - 1)]

    for iter_idx in range(n_iters):
        output = pretrained(fake_x)
        fake_label = F.softmax(output, dim=1)

        # R_prior losses
        loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
        # feature loss
        loss_bn = sum([mod.r_feature * rescale[idx] for idx, mod in enumerate(loss_r_feature_layers)]) / len(
            loss_r_feature_layers)
        # dice loss
        dice_loss = loss_fn(output, mask)
        # total loss
        loss = dice_loss * dice_weight + loss_bn * gen_loss_weight['loss_bn'] + loss_var_l1 * \
               gen_loss_weight['loss_var_l1'] + loss_var_l2 * gen_loss_weight['loss_var_l2']

        pretrained.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pid == 0:
            print(f"process {pid}: {iter_idx + 1}/{n_iters}| L1: {loss_var_l1.item():.2f}|"
                  f" L2: {loss_var_l2.item():.2f}| Batch_Norm:{loss_bn.item():.2f}| dice: {dice_loss.item()}",
                  end='\r')

        if iter_idx % 100 == 0:
            img_cnt = num.value + pid
            fake_x_iter = fake_x.detach().cpu().numpy()
            fake_label_iter = fake_label.detach().cpu().numpy()
            save_nyp(img_cnt, fake_x_iter, fake_label_iter, root_p, 'iter' + str(iter_idx))

        # log
        l1_log = loss_var_l1.item() * gen_loss_weight['loss_var_l1']
        l2_log = loss_var_l2.item() * gen_loss_weight['loss_var_l2']
        bn_log = loss_bn.item() * gen_loss_weight['loss_bn']
        dice_log = dice_loss.item() * dice_weight

        gen_loss_log['L1'].append(l1_log)
        gen_loss_log['L2'].append(l2_log)
        gen_loss_log['Batch_Norm'].append(bn_log)
        gen_loss_log['Dice'].append(dice_log)

    # unhook pretrained model
    for mod in loss_r_feature_layers:
        mod.close()

    # save image
    organ_pixels = torch.count_nonzero(torch.argmax(fake_label, dim=1), dim=(0, 1, 2, 3)).item()
    lock.acquire()

    # log
    gen_l1_loss.append(gen_loss_log['L1'])
    gen_l2_loss.append(gen_loss_log['L2'])
    gen_bn_loss.append(gen_loss_log['Batch_Norm'])
    gen_dice_loss.append(gen_loss_log['Dice'])

    if num.value < n_imgs:
        img_cnt = num.value
        num.value += 1
        lock.release()
        fake_x = fake_x.detach().cpu().numpy()
        fake_label = fake_label.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        save_nyp_mask(img_cnt, fake_x, fake_label, mask, root_p, 'final')
    else:
        lock.release()

    print(f'process {pid} ratio of foreground: {((organ_pixels / pixels) * 100):.5f}')


def gen_img_args():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument("--train_type", type=str, default='di_mask dice_loss_1')
    parser.add_argument("--gpu", type=str, default='4,5')
    parser.add_argument("--gen_epochs", type=int, default=2000)
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dice", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=1234)
    return parser


if __name__ == '__main__':
    # parser
    gen_parser = gen_img_args()
    args = gen_parser.parse_args()
    print(args)

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    mp.set_start_method('spawn')
    num = mp.Value('i', 0)
    gen_l1_loss = mp.Manager().list()
    gen_l2_loss = mp.Manager().list()
    gen_bn_loss = mp.Manager().list()
    gen_dice_loss = mp.Manager().list()

    lock = mp.Lock()
    gen_img_mask(args, num=num, gen_l1_loss=gen_l1_loss, gen_l2_loss=gen_l2_loss, gen_bn_loss=gen_bn_loss,
                 gen_dice_loss=gen_dice_loss, lock=lock)
