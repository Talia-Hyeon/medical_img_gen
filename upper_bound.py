import os
import random
from time import time
import argparse

import numpy as np
import torch.backends.cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.flare21 import FLAREDataSet, index_organs
from data.one_organ import BinaryDataSet
from loss_functions.score import *
from model.unet3D import UNet3D
from util import load_model, save_model, my_collate


def train_upperbound(args):
    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    num_epochs = args.epoch
    each_batch_size = args.batch_size // 5
    n_classes = args.num_classes
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
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[30, 60, 90, 120, 150, 180], gamma=0.5)

    # check resume & define model
    if args.resume == True:
        model, checkpoint = load_model(n_classes, check_point=True)
        # load define optimizer, lr_scheduler, start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('pretrained model is loaded!')

    else:
        start_epoch = 0
    model = nn.DataParallel(model).to(device)

    # loss function
    binary_loss_fn = SupervisedLoss(num_classes=n_classes)
    flare_loss_fn = DiceLoss(num_classes=n_classes)

    # data loader
    liver_data = BinaryDataSet(task_id=1)
    liver_loader = DataLoader(liver_data, batch_size=each_batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate)
    kidney_data = BinaryDataSet(task_id=2)
    kidney_loader = DataLoader(kidney_data, batch_size=each_batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=my_collate)
    spleen_data = BinaryDataSet(task_id=3)
    spleen_loader = DataLoader(spleen_data, batch_size=each_batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=my_collate)
    pancreas_data = BinaryDataSet(task_id=4)
    pancreas_loader = DataLoader(pancreas_data, batch_size=each_batch_size, shuffle=True,
                                 num_workers=num_workers, collate_fn=my_collate)
    flare_path = './dataset/FLARE_Dataset'
    flare_train = FLAREDataSet(root=flare_path, split='train')
    flare_loader = DataLoader(dataset=flare_train, batch_size=each_batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate)

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

        for train_iter, (flare_pack, liver_pack, kidney_pack, spleen_pack, pancreas_pack) in enumerate(
                zip(flare_loader, liver_loader, kidney_loader, spleen_loader, pancreas_loader)):

            flare_img = flare_pack['image']
            flare_img = torch.tensor(flare_img)
            flare_label = flare_pack['label']
            flare_label = torch.tensor(flare_label).to(device)

            liver_img = liver_pack['image']
            liver_img = torch.tensor(liver_img)
            liver_label = liver_pack['label']
            liver_label = torch.tensor(liver_label)

            kidney_img = kidney_pack['image']
            kidney_img = torch.tensor(kidney_img)
            kidney_label = kidney_pack['label']
            kidney_label = torch.tensor(kidney_label)

            spleen_img = spleen_pack['image']
            spleen_img = torch.tensor(spleen_img)
            spleen_label = spleen_pack['label']
            spleen_label = torch.tensor(spleen_label)

            pancreas_img = pancreas_pack['image']
            pancreas_img = torch.tensor(pancreas_img)
            pancreas_label = pancreas_pack['label']
            pancreas_label = torch.tensor(pancreas_label)

            concat_img = torch.cat([flare_img, liver_img, kidney_img, spleen_img, pancreas_img], dim=0)
            concat_img.to(device)

            concat_pred = model(concat_img)

            flare_pred = concat_pred[0].unsqueeze(dim=0)
            flare_loss = flare_loss_fn(flare_pred, flare_label)

            concat_pred = concat_pred[1:]
            concat_label = torch.cat([liver_label, kidney_label, spleen_label, pancreas_label], dim=0)
            concat_label = concat_label.to(device)
            binary_loss = binary_loss_fn(concat_pred, concat_label)

            loss = flare_loss + binary_loss
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

                val_loss = flare_loss_fn(pred_val, label_val)
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

        lr_scheduler.step()
        val_loss_meter.reset()

        if avg_dice >= best_avg_dice:
            best_avg_dice = avg_dice
            save_model(path=f'./save_model/{train_type}/best_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)

        elif epoch % 20 == 0:
            save_model(path=f'./save_model/{train_type}/epoch{epoch}_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)

        elif epoch == num_epochs - 1:
            save_model(path=f'./save_model/{train_type}/last_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)

        epoch_end = time()
        print('Epoch: {} | Dice: {} | Total Time: {:.4f}'.format(epoch + 1, dice_score, epoch_end - epoch_start))

    return model


def unet_args():
    parser = argparse.ArgumentParser(description="supervised learning")
    # train unet
    parser.add_argument("--epoch", type=int, default=170)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument("--type", type=str, default='upper_bound')
    parser.add_argument("--log_dir", type=str, default='./log_con')
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--resume", type=bool, default=False)
    return parser


if __name__ == '__main__':
    unet_parser = unet_args()
    args = unet_parser.parse_args()
    print(unet_parser)

    train_upperbound(args)
