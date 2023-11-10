import os
from time import time
import argparse

import torch.backends.cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.flare21 import FLAREDataSet
from data.flare21 import my_collate
from loss_functions.score import *
from data.flare21 import index_organs
from data.one_organ import BinaryDataSet
from model.unet3D import UNet3D
from util import save_model


def train_upperbound(args):
    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    num_epochs = args.epoch
    n_classes = args.num_classes
    num_workers = args.num_workers
    train_type = args.type
    logdir = args.log_dir

    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # make directory
    os.makedirs(f'./save_model/{train_type}', exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # logger
    logdir = args.log_dir + f'/{train_type}'
    writer = SummaryWriter(logdir)

    # define model
    model = UNet3D(num_classes=n_classes)
    model = nn.DataParallel(model).to(device)

    # define optimizer, lr_scheduler
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[30, 60, 90, 120, 150, 180], gamma=0.5)

    # loss function
    train_loss_fn = SupervisedLoss(num_classes=n_classes)
    train_loss_fn.to(device)
    val_loss_fn = DiceLoss(num_classes=n_classes)
    val_loss_fn.to(device)

    # data loader
    liver_data = BinaryDataSet(task_id=1)
    liver_loader = DataLoader(liver_data, batch_size=1, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate)
    kidney_data = BinaryDataSet(task_id=2)
    kidney_loader = DataLoader(kidney_data, batch_size=1, shuffle=True,
                               num_workers=num_workers, collate_fn=my_collate)
    spleen_data = BinaryDataSet(task_id=3)
    spleen_loader = DataLoader(spleen_data, batch_size=1, shuffle=True,
                               num_workers=num_workers, collate_fn=my_collate)
    pancreas_data = BinaryDataSet(task_id=4)
    pancreas_loader = DataLoader(pancreas_data, batch_size=1, shuffle=True,
                                 num_workers=num_workers, collate_fn=my_collate)

    flare_path = './dataset/FLARE21'
    flare_train = FLAREDataSet(root=flare_path, split='train', task_id=n_classes - 1)
    flare_loader = DataLoader(dataset=flare_train, batch_size=1, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate)

    flare_valid = FLAREDataSet(root=flare_path, split='val', task_id=n_classes - 1)
    valid_loader = DataLoader(dataset=flare_valid, batch_size=1, shuffle=False, num_workers=num_workers)

    # setup metrics
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    metric = ArgmaxDiceScore(num_classes=n_classes)
    best_avg_dice = -100.0

    # training
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time()
        iter_start = time()

        for train_iter, (flare_pack, liver_pack, kidney_pack, spleen_pack, pancreas_pack) in enumerate(
                zip(flare_loader, liver_loader, kidney_loader, spleen_loader, pancreas_loader)):

            flare_img = flare_pack['image']
            flare_img = torch.tensor(flare_img)
            flare_label = flare_pack['label']
            flare_label = torch.tensor(flare_label)

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
            concat_label = torch.cat([flare_label, liver_label, kidney_label, spleen_label, pancreas_label], dim=0)
            concat_label.to(device)
            concat_pred = model(concat_img)

            loss = train_loss_fn(concat_pred, concat_label)
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

                val_loss = val_loss_fn(pred_val, label_val)
                val_loss_meter.update(val_loss.item())

                iter_dice = metric(pred_val, label_val)
                dice_list.append(iter_dice)

            total_dice = torch.cat(dice_list, dim=0)
            dice_score = torch.mean(total_dice, dim=0)  # mean of all batches
            avg_dice = torch.mean(dice_score).item()  # mean of all organs

            # logger
            loss_report['val_loss'] = val_loss_meter.avg
            writer.add_scalars('Train/Val Loss', loss_report, args.global_epoch)

            organ_num = dice_score.shape[0]
            score_report = dict()
            for idx in range(organ_num):
                score_report[index_organs[idx + 1]] = dice_score[idx]
            writer.add_scalars('Dice Score Per Organ', score_report, args.global_epoch)
            writer.add_scalar('Avg Dice Score', avg_dice, args.global_epoch)

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
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4')
    parser.add_argument("--type", type=str, default='upper_bound')
    parser.add_argument("--log_dir", type=str, default='./log_con')
    parser.add_argument("--random_seed", type=int, default=1234)
    return parser


if __name__ == '__main__':
    unet_parser = unet_args()
    args = unet_parser.parse_args()
    print(unet_parser)

    train_upperbound(args)
