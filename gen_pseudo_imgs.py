import argparse
import os
import sys

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import unet3D_singlehead_bn_organ

from data.MOTSDataset_btcv import my_collate
from data.MOTSDataset_distill import MOTSDataSet
import timeit
from torch.utils.tensorboard import SummaryWriter  # tensorboardX

import nibabel as nib
from apex.apex import amp

start = timeit.default_timer()


#
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
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3, 4])
        var = input[0].permute(1, 0, 2, 3, 4).contiguous().view([nch, -1]).var(1, unbiased=False)
        # print(nch)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) / nch + \
                    torch.norm(module.running_mean.data - mean, 2) / nch
        # r_feature = torch.norm(module.running_var.data - var, 2) + \
        #     torch.norm(module.running_mean.data - mean, 2)

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

    # d7 = img[:, :, :-1, :-1, 1:] - img[:, :, 1:, 1:, :-1]
    # d8 = img[:, :, :-1, 1:, :-1] - img[:, :, 1:, :-1, 1:]
    # d9 = img[:, :, 1:, :-1, :-1] - img[:, :, :-1, 1:, 1:]

    # diff1 = img[:, :, :, :-1] - img[:, :, :, 1:]
    # diff2 = img[:, :, :-1, :] - img[:, :, 1:, :]
    # diff3 = img[:, :, 1:, :-1] - img[:, :, :-1, 1:]
    # diff4 = img[:, :, :-1, :-1] - img[:, :, 1:, 1:]
    # set_trace()
    # diff_list = [d1,d2,d3]
    diff_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]
    loss_var_l2 = sum([torch.norm(e) for e in diff_list])
    loss_var_l1 = sum([e.abs().mean() for e in diff_list])

    return loss_var_l1, loss_var_l2


def get_image_prior_losses_v1(inputs_jit):
    # COMPUTE total variation regularization loss
    d1 = inputs_jit[:, :, :, :, :-1] - inputs_jit[:, :, :, :, 1:]
    d2 = inputs_jit[:, :, :, :-1, :] - inputs_jit[:, :, :, 1:, :]
    d3 = inputs_jit[:, :, :-1, :, :] - inputs_jit[:, :, 1:, :, :]

    d4 = inputs_jit[:, :, :, 1:, 1:] - inputs_jit[:, :, :, :-1, :-1]
    d5 = inputs_jit[:, :, 1:, :, 1:] - inputs_jit[:, :, :-1, :, :-1]
    d6 = inputs_jit[:, :, 1:, 1:, :] - inputs_jit[:, :, :-1, :-1, :]
    # d7 = inputs_jit[:, :, :, :-1, 1:] - inputs_jit[:, :, :, 1:, :-1]
    # d8 = inputs_jit[:, :, :-1, :, 1:] - inputs_jit[:, :, 1:, :, :-1]
    # d9 = inputs_jit[:, :, :-1, 1:, :] - inputs_jit[:, :, 1:, :-1, :]

    # d10 = inputs_jit[:, :, :-1, :-1, :-1] - inputs_jit[:, :, 1:, 1:, 1:]

    d7 = inputs_jit[:, :, :-1, :-1, 1:] - inputs_jit[:, :, 1:, 1:, :-1]
    d8 = inputs_jit[:, :, :-1, 1:, :-1] - inputs_jit[:, :, 1:, :-1, 1:]
    d9 = inputs_jit[:, :, 1:, :-1, :-1] - inputs_jit[:, :, :-1, 1:, 1:]

    # diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    # diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    # diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    # diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    # set_trace()
    # print(locals().keys())
    # print(locals()['diff1'])
    # diff_list = [locals()[f'diff{i}'] for i in range(1,13+1)]
    diff_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
    loss_var_l2 = sum([torch.norm(e) for e in diff_list])
    loss_var_l1 = sum([e.abs().mean() for e in diff_list])

    # loss_var_l2 = torch.norm(diff1) + \
    #             torch.norm(diff2) + \
    #             torch.norm(diff3) + \
    #             torch.norm(diff4) + \
    #             torch.norm(diff5) + \
    #             torch.norm(diff6) + \
    #             torch.norm(diff7)

    # loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + \
    #                 (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean() + \
    #                 (diff5.abs() / 255.0).mean() + (diff6.abs() / 255.0).mean() + \
    #                 (diff7.abs() / 255.0).mean()

    # loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


def save_nii(a, p):
    nibimg = nib.Nifti1Image(a, np.eye(4) * 1)
    nib.save(nibimg, p)


def get_arguments():
    parser = argparse.ArgumentParser(description="unet3D_multihead")
    parser.add_argument("--data_dir", type=str, default='../dataset/')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--itrs_each_epoch", type=int, default=250)

    parser.add_argument("--sample_dir", type=str, default='sample', help="")
    parser.add_argument("--img_type", type=str, default='noise', help="")
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

    # federated arguments
    parser.add_argument('--num_users', type=int, default=7, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_min_update', type=int, default=80, help="the number of local update")
    parser.add_argument('--local_bs', type=int, default=2, help="local batch size: B")
    parser.add_argument('--transition_step', type=int, default=1000, help="")
    parser.add_argument('--n_sample', type=int, default=1, help="")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=str2bool, default=True, help="learning rate decay")
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

    pretrained_dir = "2000epoch"  # "roi06_ce10_2500"
    os.makedirs(f"./sample/{args.sample_dir}", exist_ok=True)
    cnt = 0
    n_runs = args.num_imgs // args.batch_size
    n_iters = args.num_epochs  # 00
    tid = args.target_task

    if args.img_type == 'noise':  # generate noise inputs
        print("generate noisy images....")
        for i in range(args.num_imgs):
            noise_input = torch.rand(list(input_size)).numpy() * 2 - 1
            save_nii(noise_input, f"./sample/{args.sample_dir}/noise_{i}img.nii.gz")
            print(f"{i}/{args.num_imgs}", end='\r')
        exit(0)

    ### load pretrained local models ###
    print("Generate pseudo images using pretrained models.")
    path = f"./pretrained/{pretrained_dir}/t{tid}.pth"
    print(f"Loading checkpoint {path}")
    pretrained = unet3D_singlehead_bn_organ.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pretrained.load_state_dict(checkpoint['model'], strict=False)

    loss_r_feature_layers = []
    for module in pretrained.modules():
        if isinstance(module, nn.BatchNorm3d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    pretrained.to(device)
    pretrained.eval()

    if args.img_type == 'di':  # deep inversion
        for i_run in range(n_runs):
            fake_x = torch.randn([args.batch_size, 1] + list(input_size), requires_grad=True, device="cuda")
            # fake_x = torch.rand([args.batch_size, 1]+list(input_size), requires_grad=True, device="cuda")
            label_prob = torch.tensor([[0.5, 0.5]] * args.batch_size, requires_grad=False, device="cuda")
            print(fake_x.is_leaf)
            optimizer = torch.optim.Adam([fake_x], lr=0.1)
            # optimizer = torch.optim.SGD([fake_x], lr=0.00001, momentum=args.momentum, )
            # optimizer = torch.optim.SGD([fake_x], lr=args.learning_rate, momentum=args.momentum, )
            if args.FP16:
                print("Note: Using FP16 during training************")
                pretrained, optimizer = amp.initialize(pretrained, optimizer, opt_level="O1")

            for iter_idx in range(n_iters):
                lr = adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)
                optimizer.zero_grad()
                pretrained.zero_grad()
                output, feat = pretrained(fake_x, None)
                prob = torch.sigmoid(output)
                # overlap = torch.sum(prob[:,0]*prob[:,1])
                # prob_avg = prob.mean(dim=(2,3,4))
                # frac = torch.mean((prob_avg-label_prob)**2)*d*h*w

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
                # R_feature loss
                rescale = [10] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                # rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
                loss_bn = sum(bn_diff) / len(loss_r_feature_layers)
                # loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # loss = loss_bn + loss_var_l2*1000 + frac
                loss = loss_bn * 10 + loss_var_l1 * 10 + loss_var_l2 * 1  # + frac

                if args.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f},", end='\r')
                # print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f}, {frac:.2f}, {overlap:.1f}, {np.round(prob_avg[0].detach().cpu().numpy(),4)}", end='\n')

            # set_trace()
            fake_x = fake_x.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            for img_idx in range(args.batch_size):
                save_preds(args, cnt, tid, fake_x, prob, img_idx)
                # save_nii(prob[img_idx, 0], f"./sample/{args.sample_dir}/task{tid}_{cnt}organ.nii.gz")
                # save_nii(prob[img_idx, 1], f"./sample/{args.sample_dir}/task{tid}_{cnt}tumor.nii.gz")
                print(f"img{cnt} is saved.")
                cnt += 1

    if args.img_type == 'di_prop':  # deep inversion
        for i_run in range(n_runs):
            fake_x = torch.randn([args.batch_size, 1] + list(input_size), requires_grad=True, device="cuda")
            print(fake_x.is_leaf)
            label_prob = torch.tensor([0.5], requires_grad=False, device="cuda")
            # label_prob = torch.tensor([[0.3]]*args.batch_size, requires_grad=False, device="cuda")
            optimizer = torch.optim.Adam([fake_x], lr=0.1)
            if args.FP16:
                print("Note: Using FP16 during training************")
                pretrained, optimizer = amp.initialize(pretrained, optimizer, opt_level="O1")

            for iter_idx in range(n_iters):
                adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)
                optimizer.zero_grad()
                pretrained.zero_grad()
                output, feat = pretrained(fake_x, None)
                prob = torch.sigmoid(output)
                # overlap = torch.sum(prob[:,0]*prob[:,1])
                prob_avg = prob.mean(dim=(2, 3, 4))
                frac = torch.mean((prob_avg.mean(dim=0) - label_prob) ** 2)
                # frac = torch.mean((prob_avg-label_prob)**2)#*d*h*w

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
                # R_feature loss
                rescale = [10] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                # rescale = [1000] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                # rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
                loss_bn = sum(bn_diff) / len(loss_r_feature_layers)
                # loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # loss = loss_bn + loss_var_l2*1000 + frac
                # loss = loss_bn*10 + loss_var_l1*10 + loss_var_l2*1 + frac
                loss = loss_bn * 10 + loss_var_l1 * 10 + loss_var_l2 * 1 + frac * 10000

                if args.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                print(
                    f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, BN{loss_bn:.2f}, frac{frac:.4f}",
                    end='\r')
                # print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f}, {frac:.2f}, {overlap:.1f}, {np.round(prob_avg[0].detach().cpu().numpy(),4)}", end='\n')

            # set_trace()
            fake_x = fake_x.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            for img_idx in range(args.batch_size):
                save_preds(args, cnt, tid, fake_x, prob, img_idx)
                cnt += 1

    if args.img_type == 'di_feature':
        train_dataset = MOTSDataSet(args.data_dir, \
                                    args.train_list,
                                    # max_iters=args.itrs_each_epoch * 1,
                                    crop_size=input_size,
                                    scale=args.random_scale,
                                    mirror=args.random_mirror,
                                    target_task=2)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   sampler=None,
                                                   collate_fn=my_collate)

        tr_loader = iter(train_loader)

        for i_run in range(n_runs):
            real_x = torch.from_numpy(next(tr_loader)['image']).cuda()
            with torch.no_grad():
                real_outputs, real_fts = pretrained(real_x, None)

            fake_x = torch.randn([args.batch_size, 1] + list(input_size), requires_grad=True, device="cuda")
            print(fake_x.is_leaf)
            label_prob = torch.tensor([[0.3]] * args.batch_size, requires_grad=False, device="cuda")
            # optimizer = torch.optim.Adam([real_x], lr=0.1)
            optimizer = torch.optim.Adam([fake_x], lr=0.1)
            if args.FP16:
                print("Note: Using FP16 during training************")
                pretrained, optimizer = amp.initialize(pretrained, optimizer, opt_level="O1")

            for iter_idx in range(n_iters):
                adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)
                optimizer.zero_grad()
                pretrained.zero_grad()
                outputs, fake_fts = pretrained(real_x, None)
                # outputs, fake_fts = pretrained(fake_x, None)
                prob = torch.sigmoid(outputs)
                prob_avg = prob.mean(dim=(2, 3, 4))
                # frac = torch.mean((prob_avg-label_prob)**2)*d*h*w
                # set_trace()
                frac = F.mse_loss(outputs, real_outputs)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
                # R_feature loss
                rescale = [10] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                # rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
                loss_bn = sum(bn_diff) / len(loss_r_feature_layers)
                # loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # feature loss
                # avg_real_ft = real_fts.sum(0)/real_fts.shape[0]
                # avg_fake_ft = fake_fts.sum(0)/fake_fts.shape[0]
                ft_loss = torch.norm(real_fts - fake_fts, 2)

                # loss = loss_bn + loss_var_l2*1000 + frac
                # loss = loss_var_l1*1000 + loss_var_l2*100 + ft_loss*100 + frac*100
                loss = loss_bn + loss_var_l1 * 1000 + loss_var_l2 * 100  # + ft_loss*100 + frac

                if args.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                print(
                    f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f},  BN{loss_bn:.2f}, frac{frac:.2f}, ft{ft_loss:.4f}", \
                    end='\n')
                # print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f}, {frac:.2f}, {overlap:.1f}, {np.round(prob_avg[0].detach().cpu().numpy(),4)}", end='\n')
                # if True:
                #     if iter_idx == 0:
                #         imgs = make_grid(real_x[:,:,32,...], normalize=True)
                #         writer.add_image('real_img', imgs, iter_idx)
                #     if iter_idx%100 == 0:
                #         imgs = make_grid(fake_x[:,:,32,...], normalize=True)
                #         # imgs = make_grid(real_x[:,:,32,...], normalize=True)
                #         writer.add_image('img', imgs, iter_idx)
                #     if iter_idx%10 == 0:
                #         writer.add_scalar('tv1', loss_var_l1, iter_idx)
                #         writer.add_scalar('tv2', loss_var_l2, iter_idx)
                #         writer.add_scalar('loss_bn', loss_bn, iter_idx)
                #         writer.add_scalar('frac', frac, iter_idx)
                #         writer.add_scalar('ft', ft_loss, iter_idx)
            # set_trace()
            fake_x = fake_x.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            for img_idx in range(args.batch_size):
                save_preds(args, cnt, tid, fake_x, prob, img_idx)
                cnt += 1

    if args.img_type == 'di_real':
        train_dataset = MOTSDataSet(args.data_dir, \
                                    args.train_list,
                                    # max_iters=args.itrs_each_epoch * 1,
                                    crop_size=input_size,
                                    scale=args.random_scale,
                                    mirror=args.random_mirror,
                                    target_task=2)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   sampler=None,
                                                   collate_fn=my_collate)

        tr_loader = iter(train_loader)

        for i_run in range(n_runs):
            real_x = torch.from_numpy(next(tr_loader)['image']).cuda()
            # fake_x = torch.tensor(real_x.data, device='cuda', dtype=float, requires_grad=True)
            fake_x = real_x.clone().detach().requires_grad_(True)
            print(fake_x.is_leaf)
            label_prob = torch.tensor([[0.3]] * args.batch_size, requires_grad=False, device="cuda")
            optimizer = torch.optim.Adam([fake_x], lr=0.1)
            if args.FP16:
                print("Note: Using FP16 during training************")
                pretrained, optimizer = amp.initialize(pretrained, optimizer, opt_level="O1")

            with torch.no_grad():
                real_outputs, real_fts = pretrained(real_x, None)

            for iter_idx in range(n_iters):
                adjust_learning_rate(optimizer, iter_idx, 0.1, n_iters, args.power)
                optimizer.zero_grad()
                pretrained.zero_grad()
                outputs, fake_fts = pretrained(fake_x, None)
                # outputs, fake_fts = pretrained(fake_x, None)
                prob = torch.sigmoid(outputs)
                prob_avg = prob.mean(dim=(2, 3, 4))
                # frac = torch.mean((prob_avg-label_prob)**2)*d*h*w
                # set_trace()
                frac = F.mse_loss(outputs, real_outputs)
                # frac = 0

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(fake_x)
                # R_feature loss
                rescale = [1] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                l = 0  # [5*l:5*l+5]
                bn_diff = [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
                # set_trace()
                loss_bn = sum(bn_diff) / len(loss_r_feature_layers)
                # loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # feature loss
                # avg_real_ft = real_fts.sum(0)/real_fts.shape[0]
                # avg_fake_ft = fake_fts.sum(0)/fake_fts.shape[0]
                # ft_loss = torch.norm(real_fts - fake_fts, 2)

                # loss = loss_bn + loss_var_l2*1000 + frac
                # loss = loss_var_l1*1000 + loss_var_l2*100 + ft_loss*100 + frac*100
                # loss = frac*100 + loss_bn*0.1 + loss_var_l1*1000 + loss_var_l2*100 #+ ft_loss*100 + frac
                # loss = loss_bn + loss_var_l1*1000 + loss_var_l2*100 #+ ft_loss*100 + frac
                loss = loss_bn + frac * 10 + loss_var_l1 * 10 + loss_var_l2 * 1

                if args.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f},  BN{loss_bn:.2f}, frac{frac:.2f}", \
                      # print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f},  BN{loss_bn:.2f}, frac{frac:.2f}, ft{ft_loss:.4f}",\
                      end='\n')
                # print(f"{tid}, {iter_idx}/{n_iters}, {loss_var_l1:.2f}, {loss_var_l2:.2f}, {loss_bn:.2f}, {frac:.2f}, {overlap:.1f}, {np.round(prob_avg[0].detach().cpu().numpy(),4)}", end='\n')
                # if True:
                #     if iter_idx == 0:
                #         imgs = make_grid(real_x[:,:,32,...], normalize=True)
                #         writer.add_image('real_img', imgs, iter_idx)
                #     if iter_idx%10 == 0:
                #         imgs = make_grid(fake_x[:,:,32,...], normalize=True)
                #         # imgs = make_grid(real_x[:,:,32,...], normalize=True)
                #         writer.add_image('img', imgs, iter_idx)
                #     if iter_idx%10 == 0:
                #         writer.add_scalar('tv1', loss_var_l1, iter_idx)
                #         writer.add_scalar('tv2', loss_var_l2, iter_idx)
                #         writer.add_scalar('loss_bn', loss_bn, iter_idx)
                #         writer.add_scalar('frac', frac, iter_idx)
                # writer.add_scalar('ft', ft_loss, iter_idx)
            # set_trace()
            fake_x = fake_x.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            for img_idx in range(args.batch_size):
                save_preds(args, cnt, tid, fake_x, prob, img_idx)
                cnt += 1


def save_preds(args, cnt, tid, fake_x, prob, img_idx):
    save_nii(fake_x[img_idx, 0], f"./sample/{args.sample_dir}/task{tid}_{cnt}img.nii.gz")
    save_nii(prob[img_idx, 0], f"./sample/{args.sample_dir}/task{tid}_{cnt}pred.nii.gz")
    print(f"img{cnt} is saved.")


if __name__ == '__main__':
    main()
