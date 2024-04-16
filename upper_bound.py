import os
import random
from time import time
import argparse

import numpy as np
import torch.backends.cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from data.flare21 import FLAREDataSet, index_organs
from data.one_organ import BinaryDataSet
from loss_functions.score import *
from model.unet3D import UNet3D
from util import load_model, save_model, my_collate, task_collate


def train_upperbound(args):
    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    num_epochs = args.epoch
    n_classes = args.num_classes
    flare_batch_size = round(args.batch_size * 0.2)
    con_batch_size = round(args.batch_size * 0.8)
    num_workers = args.num_workers
    train_type = args.type
    logdir = args.log_dir

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # make directory
    os.makedirs(f'./save_model/{train_type}', exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # logger
    logdir = args.log_dir + f'/{train_type}'
    writer = SummaryWriter(logdir)

    # define model, optimizer, lr_scheduler
    model = UNet3D(num_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.9, 0.99))
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.5)

    # check resume & define model
    if args.resume == True:
        model, checkpoint = load_model(n_classes, check_point=True)
        # load define optimizer, lr_scheduler, start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('pretrained model is loaded!')

    else:
        start_epoch = 0
    model = nn.DataParallel(model).to(device)

    # loss function
    binary_loss_fn = MaskedLoss()
    # flare_ce_loss_fn = CELoss()
    flare_dice_loss_fn = DiceLoss(num_classes=n_classes)

    # data loader
    liver_data = BinaryDataSet(task_id=1)
    kidney_data = BinaryDataSet(task_id=2)
    spleen_data = BinaryDataSet(task_id=3)
    pancreas_data = BinaryDataSet(task_id=4)
    binary_data = ConcatDataset([liver_data, kidney_data, spleen_data, pancreas_data])
    sampler = RandomSampler(binary_data)
    binary_loader = DataLoader(dataset=binary_data, batch_size=con_batch_size, drop_last=True,
                               sampler=sampler, num_workers=num_workers, collate_fn=task_collate)

    flare_path = './dataset/FLARE_Dataset'
    flare_train = FLAREDataSet(root=flare_path, split='train')
    flare_train_loader = DataLoader(dataset=flare_train, batch_size=flare_batch_size, drop_last=True,
                                    shuffle=True, num_workers=num_workers, collate_fn=my_collate)
    flare_valid = FLAREDataSet(root=flare_path, split='val')
    valid_loader = DataLoader(dataset=flare_valid, batch_size=1, shuffle=False, num_workers=num_workers)

    # setup metrics
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    metric = ArgmaxDiceScore(num_classes=n_classes)
    best_avg_dice = -100.0

    # training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start = time()
        iter_start = time()

        for train_iter, (flare_pack, binary_pack) in enumerate(zip(flare_train_loader, binary_loader)):
            flare_img = flare_pack['image']
            flare_img = torch.tensor(flare_img).to(device)
            flare_label = flare_pack['label']
            flare_label = torch.tensor(flare_label).to(device)

            binary_img = binary_pack['image']
            binary_img = torch.tensor(binary_img).to(device)
            binary_label = binary_pack['label']
            binary_label = torch.tensor(binary_label).to(device)
            task_id = binary_pack['task_id']

            con_img = torch.cat([flare_img, binary_img], dim=0)
            con_pred = model(con_img)
            flare_pred = con_pred[:flare_batch_size]
            binary_pred = con_pred[flare_batch_size:]

            flare_dice_loss = flare_dice_loss_fn(flare_pred, flare_label)
            # flare_ce_loss = flare_ce_loss_fn(flare_pred, flare_label)
            binary_loss = binary_loss_fn(binary_pred, binary_label, task_id)
            loss = binary_loss + flare_dice_loss  # + flare_ce_loss
            train_loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_iter + 1) % 5 == 0:
                iter_end = time()
                print('Epoch: {}/{} | Iters: {} | Train loss: {:.4f} | Time: {:.4f}'.format(
                    epoch + 1, num_epochs, train_iter + 1, loss.item(), (iter_end - iter_start)))
                iter_start = time()

        # logger
        loss_report = dict()
        loss_report['train_loss'] = train_loss_meter.avg
        train_loss_meter.reset()

        with torch.no_grad():
            model.eval()
            val_start = time()
            dice_list = []
            for valid_iter, pack in enumerate(valid_loader):
                img_val = pack[0].to(device)
                label_val = pack[1].to(device)
                pred_val = model(img_val)

                val_loss = flare_dice_loss_fn(pred_val, label_val)
                # val_ce_loss = flare_ce_loss_fn(pred_val, label_val)
                # val_loss = val_dice_loss + val_ce_loss
                val_loss_meter.update(val_loss.item())

                iter_dice = metric(pred_val, label_val)
                dice_list.append(iter_dice)

            total_dice = torch.cat(dice_list, dim=0)
            dice_score = torch.mean(total_dice, dim=0)  # mean of all batches
            avg_dice = torch.mean(dice_score).item()  # mean of all organs

            # logger
            loss_report['val_loss'] = val_loss_meter.avg
            writer.add_scalars('Train/Val Loss', loss_report, epoch)

            organ_num = dice_score.shape[0]
            score_report = dict()
            for idx in range(organ_num):
                score_report[index_organs[idx + 1]] = dice_score[idx]
            writer.add_scalars('Dice Score Per Organ', score_report, epoch)
            writer.add_scalar('Avg Dice Score', avg_dice, epoch)

            val_end = time()
            print('Epoch: {} | Valid loss: {:.4f} | Time: {:.4f}'.format(
                epoch + 1, val_loss.item(), (val_end - val_start)))

        # lr_scheduler.step()
        val_loss_meter.reset()

        if avg_dice >= best_avg_dice:
            best_avg_dice = avg_dice
            save_model(path=f'./save_model/{train_type}/best_model.pth',
                       model=model, optim=optimizer, epoch=epoch)

        elif epoch % 20 == 0:
            save_model(path=f'./save_model/{train_type}/epoch{epoch}_model.pth',
                       model=model, optim=optimizer, epoch=epoch)

        elif epoch == num_epochs - 1:
            save_model(path=f'./save_model/{train_type}/last_model.pth',
                       model=model, optim=optimizer, epoch=epoch)

        epoch_end = time()
        print('Epoch: {} | Dice: {} | Total Time: {:.4f}'.format(epoch + 1, dice_score, epoch_end - epoch_start))

    return model


def unet_args():
    parser = argparse.ArgumentParser(description="supervised learning")
    # train unet
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument("--type", type=str, default='upper_bound')
    parser.add_argument("--log_dir", type=str, default='./log_upperbound')
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--resume", type=bool, default=False)
    return parser


if __name__ == '__main__':
    unet_parser = unet_args()
    args = unet_parser.parse_args()
    print(unet_parser)

    train_upperbound(args)
