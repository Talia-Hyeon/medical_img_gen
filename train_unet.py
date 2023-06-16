import os
from time import time
import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.flare21 import FLAREDataSet
from data.flare21 import my_collate
from data.pseudo_img import FAKEDataSet
from model.unet3D import UNet3D
from loss_functions.score import *


def get_args():
    parser = argparse.ArgumentParser(description="train_pretrained_UNet")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='1,2,3,4')
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--pretrained_epoch", type=int, default=0)
    return parser


def draw_loss_plot(epoch_l, train_loss_l, val_loss_l):
    plt.plot(epoch_l, train_loss_l, 'ro--', label='train')
    plt.plot(epoch_l, val_loss_l, 'bo--', label='validation')
    plt.title(f'Real data Epoch: {epoch_l[-1]}')
    # plt.title('Fake data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(f'./fig/pretrained_loss.png')
    # plt.savefig(f'./fig/fake_loss.png')
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
    os.makedirs('./save_model', exist_ok=True)
    os.makedirs('./fig', exist_ok=True)

    parser = get_args()
    print(parser)
    args = parser.parse_args()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_classes = args.num_classes

    # define model, optimizer, lr_scheduler
    model = UNet3D(num_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=0.0001
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

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
    train_path = './dataset/FLARE21'
    train_data = FLAREDataSet(root=train_path, split='train', task_id=4)
    valid_data = FLAREDataSet(root=train_path, split='val', task_id=4)

    train_loader = DataLoader(dataset=train_data, batch_size=3, shuffle=True,
                              num_workers=4, collate_fn=my_collate)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=4)

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
    metric = DiceScore()
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

            if (train_iter + 1) % 10 == 0:
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
