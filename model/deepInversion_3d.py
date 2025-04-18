import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.util import truncate


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
        self.loss = nn.CrossEntropyLoss(reduce='mean')  # calulate per batch loss
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

        gt = torch.tensor([[0.4, 0.2, 0.15, 0.15, 0.1]])[:, :self.num_classes]
        gt = F.softmax(gt, dim=1).to(x.device) * torch.ones_like(x)
        return self.loss(x, gt)


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr * ((1 - float(i_iter) / num_stemps) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def save_nyp(cnt, fake_x, fake_label, root, name):
    # normalization
    fake_x = truncate(fake_x)

    # add channel & probability to binary
    fake_label = np.argmax(fake_label, axis=1)[np.newaxis, :]

    # save
    save_img_path = osp.join(root, 'img', name + '.npy')
    save_label_path = osp.join(root, 'mask', name + '.npy')

    np.save(save_img_path, fake_x)
    np.save(save_label_path, fake_label)
    print(f"img{cnt} is saved.")


def save_nyp_mask(cnt, fake_x, fake_label, real_label, root, name):
    # normalization
    fake_x = truncate(fake_x)

    # add channel & probability to binary
    fake_label = np.argmax(fake_label, axis=0)[np.newaxis, :]
    real_label = np.argmax(real_label, axis=0)[np.newaxis, :]

    # save
    save_img_path = osp.join(root, 'img', name + '_' + str(cnt) + '.npy')
    save_label_path = osp.join(root, 'mask', name + '_' + str(cnt) + '.npy')
    save_real_label_path = osp.join(root, 'label', name + '_' + str(cnt) + '.npy')

    np.save(save_img_path, fake_x)
    np.save(save_label_path, fake_label)
    np.save(save_real_label_path, real_label)
    print(f"img{cnt}_iter{iter} is saved.")
