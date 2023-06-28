import os
from time import time
import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from data.flare21 import FLAREDataSet
from data.flare21 import my_collate
from data.btcv import BTCVDataSet
from data.pseudo_img import FAKEDataSet
from model.unet3D import UNet3D
from loss_functions.score import *


def get_args():
    parser = argparse.ArgumentParser(description="train_pretrained_UNet")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--task_id", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument("--pretrained_model", type=str, default=None)
    return parser


def draw_loss_plot(epoch_l, train_loss_l, val_loss_l):
    plt.plot(epoch_l, train_loss_l, 'ro--', label='train')
    plt.plot(epoch_l, val_loss_l, 'bo--', label='validation')
    plt.title(f'Real data Epoch: {epoch_l[-1]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(f'./fig/pretrained_loss.png')
    plt.close()


def save_model(path, model, optim, lr_sch, epoch):
    net_state = model.module.state_dict()
    states = {
        'model': net_state,
        'optimizer': optim.state_dict(),
        'scheduler': lr_sch.state_dict(),
        'epoch': epoch + 1
    }
    torch.save(states, path)


def main():
    parser = get_args()
    print(parser)
    args = parser.parse_args()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    n_classes = args.num_classes
    task_id = args.task_id
    batch_size = args.batch_size
    num_workers = args.num_workers

    # make directory
    os.makedirs('./save_model', exist_ok=True)
    os.makedirs('./fig', exist_ok=True)

    # define model, optimizer, lr_scheduler
    model = UNet3D(num_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5, betas=(0.9, 0.99))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 80, 100, 120], gamma=0.5)

    # resume
    if args.pretrained_model != None:
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('pretrained model is loaded!')

    else:
        start_epoch = 0
    num_epochs = args.epoch + start_epoch
    model = nn.DataParallel(model).to(device)

    # loss function
    loss_function = CELoss(num_classes=n_classes)
    loss_function.to(device)

    # real data loader
    flared_path = './dataset/FLARE21'
    # btcv_path = './dataset/BTCV/Trainset'
    flared_train = FLAREDataSet(root=flared_path, split='train', task_id=task_id)
    flared_valid = FLAREDataSet(root=flared_path, split='val', task_id=task_id)
    # btcv_train = BTCVDataSet(root=btcv_path, split='train', task_id=task_id)
    # btcv_val = BTCVDataSet(root=btcv_path, split='val', task_id=task_id)

    # train_data = ConcatDataset([flared_train, btcv_train])
    # val_data = ConcatDataset([flared_valid, btcv_val])
    train_loader = DataLoader(dataset=flared_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate)
    valid_loader = DataLoader(dataset=flared_valid, batch_size=1, shuffle=False, num_workers=num_workers)

    # # fake data loader
    # train_path = './sample'
    # train_data = FAKEDataSet(root=train_path, split='train', task_id=4)
    # valid_data = FAKEDataSet(root=train_path, split='val', task_id=4)
    #
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
    #                           num_workers=4, collate_fn=my_collate)
    # valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=4)

    # setup metrics
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    metric = DiceScore(num_classes=n_classes)
    best_avg_dice = -100.0

    # training
    train_loss_l = []
    val_loss_l = []
    epoch_l = []
    for epoch in range(start_epoch, num_epochs):

        epoch_start = time()
        iter_start = time()

        for train_iter, pack in enumerate(train_loader):
            img = pack['image']
            img = torch.tensor(img).to(device)
            label = pack['label']
            label = torch.tensor(label).to(device)

            pred = model(img)
            loss = loss_function(pred, label)
            train_loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_iter + 1) % 5 == 0:
                iter_end = time()
                print('Epoch: {}/{} | Iters: {} | Train loss: {:.4f} | Time: {:.4f}'.format(
                    epoch + 1, num_epochs, train_iter + 1, loss.item(), (iter_end - iter_start)))
                # newly check iter. start time
                iter_start = time()

        train_loss_l.append(train_loss_meter.avg)
        epoch_l.append(epoch)
        train_loss_meter.reset()

        with torch.no_grad():
            model.eval()
            val_start = time()
            dice_list = []
            for valid_iter, pack in enumerate(valid_loader):
                img_val = pack[0].to(device)
                label_val = pack[1].to(device)
                pred_val = model(img_val)

                val_loss = loss_function(pred_val, label_val)
                val_loss_meter.update(val_loss.item())

                iter_dice = metric(pred_val, label_val)
                dice_list.append(iter_dice)

            val_loss_l.append(val_loss_meter.avg)
            total_dice = torch.stack(dice_list)
            dice_score = torch.mean(total_dice, dim=0)
            avg_dice = torch.mean(dice_score).item()

            val_end = time()
            print('Epoch: {} | Valid loss: {:.4f} | Time: {:.4f}'.format(
                epoch + 1, val_loss.item(), (val_end - val_start)))

        lr_scheduler.step()
        val_loss_meter.reset()

        if avg_dice >= best_avg_dice:
            best_avg_dice = avg_dice
            best_dice = dice_score
            save_model(path=f'./save_model/{epoch}_best_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)
            # torch.save(model.state_dict(), f'./save_model/best_model_fake.pth')

        elif epoch % 20 == 0:
            save_model(path=f'./save_model/{epoch}_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)
            # torch.save((model.state_dict(), f'./save_model/{epoch}_model_fake.pth'))
            draw_loss_plot(epoch_l, train_loss_l, val_loss_l)

        elif epoch == num_epochs - 1:
            save_model(path=f'./save_model/{epoch}_last_model.pth',
                       model=model, optim=optimizer, lr_sch=lr_scheduler, epoch=epoch)
            # torch.save((model.state_dict(), f'./save_model/last_model_fake.pth'))

        epoch_end = time()
        print('Epoch: {} | Best Dice: {} | Total Time: {:.4f}'.format(epoch + 1, best_dice, epoch_end - epoch_start))


if __name__ == '__main__':
    main()
