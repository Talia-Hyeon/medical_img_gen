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

from model.deepInversion_3d import DeepInversionFeatureHook, get_image_prior_losses, save_nyp
from data.flare21_gen import FLARE_Mask
from loss_functions.score import DiceLoss
from util import load_model


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

    device_ids = [i for i in range(torch.cuda.device_count())]
    input_size = (40, 270, 220)
    pixels = input_size[0] * input_size[1] * input_size[2]

    cudnn.benchmark = True

    # make directory
    root_p = f"./sample/{train_type}/"
    # logger
    logdir = args.log_dir + f'/{args.train_type}'

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
        fake_xs = [torch.randn([1, 1] + list(input_size), requires_grad=True, device=device_ids[i]) for i in
                   range(len(device_ids))]
        optimizers = [torch.optim.Adam([fake_xs[i]], lr=0.1) for i in range(len(device_ids))]

        # define the mask
        mask, name = next(real_dataloader_infinite_iter)

        children = list()

        for pid in range(len(device_ids)):
            child = mp.Process(target=gen_img, args=(
                pid, pretraineds[pid], fake_xs[pid], optimizers[pid], mask[pid].to(device_ids[pid]), name[pid],
                loss_fns[pid], root_p, logdir, pixels, n_iters, n_imgs, num, lock))
            children.append(child)
            child.start()

        for child in children:
            child.join()

        cnt = num.value

        end = timeit.default_timer()
        print(f'generataion end: {(end - start):.3f} s')


def gen_img(pid, pretrained, fake_x, optimizer, mask, name, loss_fn,
            root_p, logdir, pixels, n_iters, n_imgs, num, lock):
    # make dirs
    root_p = root_p + '/' + name
    os.makedirs(f'{root_p}/img', exist_ok=True)
    os.makedirs(f'{root_p}/mask', exist_ok=True)

    # log
    logdir = logdir + '/' + name
    writer = SummaryWriter(logdir)

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
        loss = dice_loss * 1e-2 + loss_bn * 1 + loss_var_l1 * 2.5e-5 + loss_var_l2 * 3e-8

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
        loss_report = dict()
        loss_report['L1'] = loss_var_l1.item()
        loss_report['L2'] = loss_var_l2.item()
        loss_report['Batch_Norm'] = loss_bn.item()
        loss_report['Dice'] = dice_loss.item()
        writer.add_scalars('Loss', loss_report, iter_idx)

        organ_report = dict()
        organ_pixels = torch.count_nonzero(torch.argmax(fake_label, dim=1), dim=(0, 1, 2, 3)).item()
        ratio_organ = (organ_pixels / pixels) * 100
        organ_report['Ratio of Organ'] = ratio_organ
        writer.add_scalars('Ratio of Organ', organ_report, iter_idx)

    # unhook pretrained model
    for mod in loss_r_feature_layers:
        mod.close()

    # save image
    organ_pixels = torch.count_nonzero(torch.argmax(fake_label, dim=1), dim=(0, 1, 2, 3)).item()
    lock.acquire()
    if num.value < n_imgs:
        img_cnt = num.value
        num.value += 1
        lock.release()
        fake_x = fake_x.detach().cpu().numpy()
        fake_label = fake_label.detach().cpu().numpy()
        save_nyp(img_cnt, fake_x, fake_label, root_p, 'final')
    else:
        lock.release()

    print(f'process {pid} ratio of foreground: {((organ_pixels / pixels) * 100):.5f}')


def gen_img_args():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument("--train_type", type=str, default='di_mask')
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument("--gen_epochs", type=int, default=2000)
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default='./log_img')
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
    lock = mp.Lock()
    gen_img_mask(args, num=num, lock=lock)
