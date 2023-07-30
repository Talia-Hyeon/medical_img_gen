import argparse
import os
import timeit
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
    loss_var_l2 = sum([torch.norm(e) for e in diff_list])
    loss_var_l1 = sum([e.abs().mean() for e in diff_list])

    return loss_var_l1, loss_var_l2


class ClassLoss(nn.Module):
    def __init__(self, r_args, num_classes):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

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
        gt = torch.ones_like(x)
        # gt[:, 0] = 0  # don't contain the background into generated class
        gt = F.softmax(gt, dim=1)

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


def save_preds(cnt, fake_x, fake_label, img_idx, root):
    save_nii(fake_x[img_idx, 0], f".{root}/Img/{cnt}img.nii.gz")
    save_nii(fake_label[img_idx], f".{root}/Pred/{cnt}pred.nii.gz")
    print(f"img{cnt} is saved.")


def get_arguments():
    parser = argparse.ArgumentParser(description="image generation")
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--num_imgs", type=int, default=50)  # 288
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0,1,2,3')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default='./save_model/epoch145_best_model.pth')
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--power", type=float, default=0.9)
    return parser


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    cnt = 0
    n_runs = args.num_imgs // args.batch_size
    n_iters = args.num_epochs

    task_id = args.task_id
    num_classes = args.num_classes
    path = args.pretrained_model
    # pre_path = f'./save_model/{task_id}/best_model.pth'
    input_size = (160, 192, 192)

    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # make directory
    root_p = f"./sample/mask"
    os.makedirs(root_p, exist_ok=True)
    os.makedirs(f"{root_p}/Img", exist_ok=True)
    os.makedirs(f"{root_p}/Pred", exist_ok=True)

    # load the pretrained model
    print("Generate pseudo images using pretrained models.")
    print(f"Loading checkpoint {path}")
    pretrained = UNet3D(num_classes=num_classes)
    checkpoint = torch.load(path)
    pretrained.load_state_dict(checkpoint['model'], strict=False)
    pretrained = nn.DataParallel(pretrained).to(device)

    # loss function
    # feature loss
    loss_r_feature_layers = []
    for module in pretrained.modules():
        if isinstance(module, nn.BatchNorm3d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # define the dataset and loss fn for the mask
    real_data = FLAREDataSet(root='./dataset/FLARE21', split='train', task_id=task_id)
    real_dataloader = data.DataLoader(dataset=real_data, batch_size=args.batch_size, shuffle=True)
    loss_fn = DiceLoss(num_classes=task_id + 1)
    loss_fn.to(device)

    # generate fake images
    pretrained.eval()

    for i_run in range(n_runs):
        fake_x = torch.randn([args.batch_size, 1] + list(input_size), requires_grad=True, device=device)
        print(fake_x.is_leaf)  # 텐서가 그래프의 말단 노드인지

        # # class loss
        # class_loss_fn = ClassLoss(r_args=[(5, 1, 0.1, 10), (5, 1, 0.1, 10),  # background, liver
        #                                   (5, 1, 0.1, 10), (5, 1, 0.1, 10), (5, 1, 0.1, 10)],  # kidney,spleen,pancreas
        #                           num_classes=num_classes)
        # class_loss_fn.to(device)

        # define the mask
        img, mask, name = next(iter(real_dataloader))
        mask = mask.to(device)

        optimizer = torch.optim.Adam([fake_x], lr=0.1)

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
            # class_loss = class_loss_fn(fake_label)
            # dice loss
            dice_loss = loss_fn(output, mask)
            # total loss
            # loss = class_loss * 1 + loss_bn * 1 + loss_var_l1 * 2.5e-5 + loss_var_l2 * 3e-8
            loss = dice_loss * 1e-2 + loss_bn * 1 + loss_var_l1 * 2.5e-5 + loss_var_l2 * 3e-8

            pretrained.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"{iter_idx}/{n_iters}| L1: {loss_var_l1:.2f}|"
                  f" L2: {loss_var_l2:.2f}| Batch_Norm:{loss_bn:.2f}| Dice: {dice_loss}", end='\r')

        fake_x = fake_x.detach().cpu().numpy()
        fake_label = fake_label.detach().cpu().numpy()
        for img_idx in range(args.batch_size):
            save_preds(cnt, fake_x, fake_label, img_idx, root_p)
            print(f"img{cnt} is saved.")
            cnt += 1


if __name__ == '__main__':
    main()
