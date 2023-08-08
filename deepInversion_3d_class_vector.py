import argparse
import os
import timeit
from itertools import cycle
import random

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib

from model.unet3D import UNet3D
from data.flare21 import FLAREDataSet
from loss_functions.score import DiceLoss

start = timeit.default_timer()


# code from https://github.com/NVlabs/DeepInversion
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        # module: real's batchnorm_layer, input: synth
        nch = input[0].shape[1]  # # of channel?
        mean = input[0].mean([0, 2, 3, 4])  # 0: batch, 1: channel, 2: depth, 3: height, 4: width
        var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence

        r_feature = torch.norm(module.running_var.data - var, 2) / nch + \
                    torch.norm(module.running_mean.data - mean, 2) / nch

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def get_image_prior_losses(img):
    # COMPUTE total variation regularization loss
    d1 = img[:, :, :, :, :-1] - img[:, :, :, :, 1:]
    d2 = img[:, :, :, :-1, :] - img[:, :, :, 1:, :]
    d3 = img[:, :, :-1, :, :] - img[:, :, 1:, :, :]
    d4 = img[:, :, :, 1:, :1] - img[:, :, :, :-1, :-1]
    d5 = img[:, :, :, 1:, :-1] - img[:, :, :, :-1, :1]
    d6 = img[:, :, 1:, :, 1:] - img[:, :, :-1, :, :-1]
    d7 = img[:, :, 1:, :, :-1] - img[:, :, :-1, :, 1:]
    d8 = img[:, :, 1:, 1:, :] - img[:, :, :-1, :-1, :]
    d9 = img[:, :, 1:, :-1, :] - img[:, :, :-1, 1:, :]
    d10 = img[:, :, :-1, :-1, 1:] - img[:, :, 1:, 1:, :-1]
    d11 = img[:, :, :-1, 1:, :-1] - img[:, :, 1:, :-1, 1:]
    d12 = img[:, :, 1:, :-1, :-1] - img[:, :, :-1, 1:, 1:]
    d13 = img[:, :, 1:, 1:, 1:] - img[:, :, :-1, :-1, :-1]

    diff_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]

    loss_var_l2 = 0.0
    loss_var_l1 = 0.0

    for e in diff_list:
        loss_var_l2 += torch.norm(e)
        loss_var_l1 += torch.mean(torch.abs(e))

    return loss_var_l1, loss_var_l2


class ClassLoss(nn.Module):
    def __init__(self, r_args, num_classes):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        r = []
        for organ in range(num_classes):
            mean, std, upper, lower = r_args[organ]
            r_k = torch.distributions.normal.Normal(torch.tensor(float(mean)), torch.tensor(float(std))).sample()
            r_k = min(torch.tensor(upper), r_k)
            r_k = max(torch.tensor(lower), r_k)
            r.append(r_k)

        r = torch.tensor(r).reshape(1, num_classes)  # add batch
        self.register_buffer('r', r)  # store the untrained value

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert C == self.r.size(1), "channel's size don't match"

        x = (1 / self.r) * torch.log(
            torch.sum(torch.exp(self.r[..., None, None, None] * x), dim=(2, 3, 4)) / (D * H * W))  # shape = (B,C)
        # gt = torch.ones_like(x)
        # # gt[:, 0] = 0  # don't contain the background into generated class
        # gt = F.softmax(gt, dim=1)

        gt = torch.tensor([[0.05, 0.35, 0.2, 0.25, 0.15]]).to(x.device) * torch.ones_like(x)
        return self.loss(x, gt)


def lr_poly(base_lr, iter_idx, max_iter, power):
    return base_lr * ((1 - float(iter_idx) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def save_nii(a, p):  # img_data, path
    nibimg = nib.Nifti1Image(a, np.eye(4) * 1)  # img.affine
    nib.save(nibimg, p)


def save_preds(cnt, fake_x, fake_label, root):
    save_nii(fake_x.numpy(), f"{root}/Img/{cnt}img.nii.gz")
    save_nii(fake_label.numpy(), f"{root}/Pred/{cnt}pred.nii.gz")
    print(f"img{cnt} is saved.")
    return


def gen_img_vector(args, device, task_id=1):
    # hyper-parameter
    cnt = 0
    n_imgs = args.num_imgs
    n_iters = args.gen_epochs
    img_check_points = [(n_iters // 10) * (i + 1) for i in range(9)]

    num_classes = args.num_classes
    batch_size = args.gen_batch_size
    pre_path = './save_model/epoch145_best_model.pth'
    input_size = (80, 96, 96)
    pixels = input_size[0] * input_size[1] * input_size[2]

    organ_percentage = 0.007

    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # make directory
    root_p = f"./sample/mask/{task_id}"
    os.makedirs(root_p, exist_ok=True)
    os.makedirs(f"{root_p}/Img", exist_ok=True)
    os.makedirs(f"{root_p}/Pred", exist_ok=True)

    # load the pretrained model
    print("Generate pseudo images using pretrained models.")
    print(f"Loading checkpoint {pre_path}")
    pretrained = UNet3D(num_classes=task_id + 1)
    checkpoint = torch.load(pre_path)
    pretrained.load_state_dict(checkpoint['model'], strict=False)
    pretrained = nn.DataParallel(pretrained).to(device)

    # loss function
    # feature loss
    loss_r_feature_layers = []
    for module in pretrained.modules():
        if isinstance(module, nn.BatchNorm3d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # generate fake images
    pretrained.eval()
    while cnt < n_imgs:
        fake_x = torch.randn([batch_size, 1] + list(input_size), requires_grad=True, device=device)

        # class loss
        class_loss_fn = ClassLoss(r_args=[(4, 1, 0.1, 7), (0.5, 0.1, 0.1, 1),  # background, liver
                                          (2, 0.4, 0.1, 4), (1, 0.3, 0.1, 3), (3, 0.5, 0.1, 5)],
                                  # kidney, spleen, pancreas
                                  num_classes=num_classes)
        class_loss_fn.to(device)

        optimizer = torch.optim.Adam([fake_x], lr=0.1)

        organ_pixels = 0
        for iter_idx in range(n_iters):
            # lr = adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)

            output = pretrained(fake_x)
            fake_label = F.softmax(output, dim=1)

            # R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
            # R_feature loss
            rescale = [10] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
            bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
            loss_bn = torch.sum(torch.tensor(bn_diff, requires_grad=True)) / len(loss_r_feature_layers)
            # class loss
            class_loss = class_loss_fn(fake_label)
            # total loss
            loss = class_loss * 1 + loss_bn * 1 + loss_var_l1 * 2.5e-5 + loss_var_l2 * 3e-8

            pretrained.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"{iter_idx + 1}/{n_iters}| L1: {loss_var_l1:.2f}|"
                  f" L2: {loss_var_l2:.2f}| Batch_Norm:{loss_bn:.2f}| class: {class_loss}", end='\r')

            # check if image meets condition
            if iter_idx in img_check_points:
                assert fake_label.shape[0] == 1, 'batch size must be 1.'
                organ_pixels = torch.count_nonzero(torch.argmax(fake_label, dim=1), dim=(1, 2, 3)).item()
                if organ_pixels / pixels >= organ_percentage:
                    break

        print()  # formatting

        # save image
        if organ_pixels / pixels >= organ_percentage and cnt < n_imgs:
            fake_x = fake_x.detach().cpu()
            fake_label = fake_label.detach().cpu()
            save_preds(cnt, fake_x[0, 0], fake_label[0], root_p)
            cnt += 1


def gen_img_args():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--gen_epochs", type=int, default=5000)
    parser.add_argument("--num_imgs", type=int, default=288)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--power", type=float, default=0.9)
    return parser


if __name__ == '__main__':
    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parser
    gen_parser = gen_img_args()
    args = gen_parser.parse_args()
    print(args)

    gen_img_vector(args, device, task_id=4)
