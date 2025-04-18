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
import pandas as pd

from model.deepInversion_3d import DeepInversionFeatureHook, get_image_prior_losses, save_nyp, save_nyp_mask
from data.flare21_gen import FLARE_Mask
from loss_functions.score import DiceLoss
from util.util import load_model
from gen_img_class_vector import image_size


def gen_img_mask(args, num, lock):
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
    num_batch = batch_size // len(device_ids)
    pixels = image_size[0] * image_size[1] * image_size[2] * num_batch

    cudnn.benchmark = True

    # make directory
    root_p = f"./sample/{train_type}/"
    os.makedirs(f'{root_p}/img', exist_ok=True)
    os.makedirs(f'{root_p}/mask', exist_ok=True)
    os.makedirs(f'{root_p}/label', exist_ok=True)

    # logger
    os.makedirs(f'{root_p}/loss_log', exist_ok=True)
    img_log_root = f"./log_img/{train_type}"

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
        fake_xs = [torch.randn([num_batch, 1] + list(image_size), requires_grad=True, device=device_ids[i]) for i in
                   range(len(device_ids))]
        optimizers = [torch.optim.Adam([fake_xs[i]], lr=0.1) for i in range(len(device_ids))]

        # define the mask
        mask, name = next(real_dataloader_infinite_iter)

        mask, name = item_concat(mask, name, batch_size, num_batch)

        children = list()

        for pid in range(len(device_ids)):
            child = mp.Process(target=gen_img, args=(
                pid, pretraineds[pid], fake_xs[pid], optimizers[pid], mask[pid].to(device_ids[pid]), name[pid],
                loss_fns[pid], dice_weight, root_p, img_log_root, pixels, num_batch, n_iters, n_imgs, num, lock))
            children.append(child)
            child.start()

        for child in children:
            child.join()

        cnt = num.value

        end = timeit.default_timer()
        print(f'generataion end: {(end - start):.3f} s')


def gen_img(pid, pretrained, fake_x, optimizer, mask, name, loss_fn,
            dice_weight, root_p, img_log_root, pixels, num_batch, n_iters, n_imgs, num, lock):
    # log
    # gen_loss_log = {'L1': [], 'L2': [], 'Batch_Norm': [], 'Dice': []}
    loss_log = []

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
        loss = dice_loss * dice_weight + loss_bn * 1 + loss_var_l1 * 2.5e-5 + loss_var_l2 * 3e-8

        pretrained.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        l1_log = loss_var_l1.item() * 2.5e-5
        l2_log = loss_var_l2.item() * 3e-8
        bn_log = loss_bn.item() * 1
        dice_log = dice_loss.item() * dice_weight

        # gen_loss_log['L1'].append(l1_log)
        # gen_loss_log['L2'].append(l2_log)
        # gen_loss_log['Batch_Norm'].append(bn_log)
        # gen_loss_log['Dice'].append(dice_log)

        if pid == 0:
            print(f"process {pid}: {iter_idx + 1}/{n_iters}| L1: {loss_var_l1.item():.2f}|"
                  f" L2: {loss_var_l2.item():.2f}| Batch_Norm:{loss_bn.item():.2f}| dice: {dice_loss.item()}",
                  end='\r')

        if iter_idx % 500 == 0:
            for i in range(num_batch):
                img_cnt = num.value + pid
                fake_x_iter = fake_x.detach().cpu().numpy()
                fake_label_iter = fake_label.detach().cpu().numpy()
                save_nyp(img_cnt, fake_x_iter[i], fake_label_iter[i], root_p, name[i] + str(iter_idx))

                loss_log.append({
                    'Iteration': iter_idx,
                    'L1': l1_log,
                    'L2': l2_log,
                    'BatchNorm': bn_log,
                    'Dice': dice_log
                })

    # unhook pretrained model
    for mod in loss_r_feature_layers:
        mod.close()

    # save image
    organ_pixels = torch.count_nonzero(torch.argmax(fake_label, dim=1), dim=(0, 1, 2, 3)).item()
    lock.acquire()

    if num.value < n_imgs:
        img_cnt = num.value
        num.value += num_batch
        lock.release()
        fake_x = fake_x.detach().cpu().numpy()
        fake_label = fake_label.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        for i in range(num_batch):
            save_nyp_mask(img_cnt, fake_x[i], fake_label[i], mask[i], root_p, name[i])
            img_cnt += 1

        df = pd.DataFrame(loss_log, columns=['Iteration', 'L1', 'L2', 'BatchNorm', 'Dice'])
        df.to_csv(f'{root_p}/loss_log/{name[i]}.csv', index=False)

        # log img gen loss
        # save_log_p = f'{img_log_root}/{name[0]}_{img_cnt}'
        # os.makedirs(save_log_p, exist_ok=True)
        # gen_log_l1 = np.array(gen_loss_log['L1'])
        # gen_log_l2 = np.array(gen_loss_log['L2'])
        # gen_log_bn = np.array(gen_loss_log['Batch_Norm'])
        # gen_log_dice = np.array(gen_loss_log['Dice'])

        # np.save(f'{save_log_p}/l1.npy', gen_log_l1)
        # np.save(f'{save_log_p}/l2.npy', gen_log_l2)
        # np.save(f'{save_log_p}/bn.npy', gen_log_bn)
        # np.save(f'{save_log_p}/dice.npy', gen_log_dice)

    else:
        lock.release()

    print(f'process {pid} ratio of foreground: {((organ_pixels / pixels) * 100):.5f}')


def item_concat(mask, name, batch_size, num_batch):
    mask_list = []
    name_list = []
    for batch_i in range(0, batch_size, num_batch):
        mask_l = []
        name_l = []
        for list_i in range(num_batch):
            mask_l.append(mask[batch_i + list_i])
            name_l.append(name[batch_i + list_i])
        con_mask = torch.stack(mask_l, dim=0)
        mask_list.append(con_mask)
        name_list.append(name_l)
    return mask_list, name_list


def gen_img_args():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument("--train_type", type=str, default='di_mask w loss log')
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument("--gen_epochs", type=int, default=2000)
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=16)
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
    # gen_l1_loss = mp.Manager().list()
    # gen_l2_loss = mp.Manager().list()
    # gen_bn_loss = mp.Manager().list()
    # gen_dice_loss = mp.Manager().list()

    lock = mp.Lock()
    # gen_img_mask(args, num=num, gen_l1_loss=gen_l1_loss, gen_l2_loss=gen_l2_loss, gen_bn_loss=gen_bn_loss,
    #              gen_dice_loss=gen_dice_loss, lock=lock)
    gen_img_mask(args, num=num, lock=lock)
