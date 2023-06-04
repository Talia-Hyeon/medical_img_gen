import argparse
import os, sys
import timeit

sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter  # tensorboardX
import nibabel as nib
from apex.apex import amp

import unet3D_singlehead_bn_organ
from data.MOTSDataset_btcv import MOTSValDataSet, my_collate
from data.MOTSDataset_distill import MOTSDataSet
from loss_functions import loss

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
        mean = input[0].mean([0, 2, 3, 4])  # 0: batch, 1: channel, 3: height, 3: width, 2: depth
        var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)
        # print(nch)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) / nch + \
                    torch.norm(module.running_mean.data - mean, 2) / nch

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def save_nii(a, p):  # img_data, path
    nibimg = nib.Nifti1Image(a, np.eye(4) * 1)  # img.affine
    nib.save(nibimg, p)


def get_arguments():
    parser = argparse.ArgumentParser(description="unet3D_multihead")
    parser.add_argument("--data_dir", type=str, default='../dataset/')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--itrs_each_epoch", type=int, default=250)

    parser.add_argument("--sample_dir", type=str, default='sample', help="")
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/fold1/')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_imgs", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--input_size", type=str, default='64,64,64')
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')

    return parser


def lr_poly(base_lr, iter_idx, max_iter, power):
    return base_lr * ((1 - float(iter_idx) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    args = parser.parse_args()
    writer = SummaryWriter(f"./snapshots/{args.sample_dir}")

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda:{}'.format(args.local_rank))

    os.makedirs(f"./sample/{args.sample_dir}", exist_ok=True)
    cnt = 0
    n_runs = args.num_imgs // args.batch_size
    n_iters = args.num_epochs  # 500
    tid = args.target_task  # 0

    ### load pretrained local models ###
    print("Generate pseudo images using pretrained models.")
    pretrained_dir = "2000epoch"
    path = f"./pretrained/{pretrained_dir}/t{tid}.pth"
    print(f"Loading checkpoint {path}")
    pretrained = unet3D_singlehead_bn_organ.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # GPU에서 저장한 모델을 CPU에서 불러오기
    pretrained.load_state_dict(checkpoint['model'], strict=False)

    loss_r_feature_layers = []
    for module in pretrained.modules():
        if isinstance(module, nn.BatchNorm3d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    pretrained.to(device)
    pretrained.eval()

    for i_run in range(n_runs):
        fake_x = torch.randn([args.batch_size, 1] + list(input_size), requires_grad=True, device="cuda")
        print(fake_x.is_leaf)
        optimizer = torch.optim.Adam([fake_x], lr=0.1)
        if args.FP16:
            print("Note: Using FP16 during training************")
            pretrained, optimizer = amp.initialize(pretrained, optimizer, opt_level="O1")

        for iter_idx in range(n_iters):
            lr = adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)
            optimizer.zero_grad()
            pretrained.zero_grad()
            output, feat = pretrained(fake_x, None)
            prob = torch.sigmoid(output)

            # R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
            # R_feature loss
            rescale = [10] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
            bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
            loss_bn = sum(bn_diff) / len(loss_r_feature_layers)
            loss = loss_bn * 10 + loss_var_l1 * 10 + loss_var_l2 * 1  # + frac

            if args.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f},", end='\r')

        fake_x = fake_x.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        for img_idx in range(args.batch_size):
            save_preds(args, cnt, tid, fake_x, prob, img_idx)
            print(f"img{cnt} is saved.")
            cnt += 1


def save_preds(args, cnt, tid, fake_x, prob, img_idx):
    save_nii(fake_x[img_idx, 0], f"./sample/{args.sample_dir}/task{tid}_{cnt}img.nii.gz")
    save_nii(prob[img_idx, 0], f"./sample/{args.sample_dir}/task{tid}_{cnt}pred.nii.gz")
    print(f"img{cnt} is saved.")


if __name__ == '__main__':
    main()
