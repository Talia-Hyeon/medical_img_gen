import os
from time import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.flare21 import FLAREDataSet
from data.flare21 import my_collate
from data.pseudo_img import FAKEDataSet
from model.unet3D import UNet3D
from loss_functions.score import *


def main():
    os.makedirs('./save_model', exist_ok=True)
    os.makedirs('./fig', exist_ok=True)
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    # Hyper-parameters
    num_epochs = 150
    n_classes = 5

    model = UNet3D(num_classes=n_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=0.0001
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_function = CELoss(num_classes=n_classes)
    loss_function.to(device)

    # real data loader
    train_path = './dataset/FLARE21'
    train_data = FLAREDataSet(root=train_path, split='train', task_id=4)
    valid_data = FLAREDataSet(root=train_path, split='val', task_id=4)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True,
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
    metrics = Score(n_classes)

    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    best_iou = -100.0

    # training
    train_loss_l = []
    val_loss_l = []
    epoch_l = []
    for epoch in range(num_epochs):

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

            if (train_iter + 1) % (len(train_loader) // 10) == 0:
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
            for valid_iter, pack in enumerate(valid_loader):
                img_val = pack[0].to(device)
                label_val = pack[1].to(device)

                pred_val = model(img_val)
                val_loss = loss_function(pred_val, label_val)
                val_loss_meter.update(val_loss.item())

                ours = torch.argmax(pred_val, dim=1).cpu().numpy()
                gt = torch.argmax(label_val, dim=1).cpu().numpy()
                metrics.update(gt, ours)

            val_loss_l.append(val_loss_meter.avg)
            val_end = time()
            print('Epoch: {} | Valid loss: {:.4f} | Time: {:.4f}'.format(
                epoch + 1, val_loss.item(), (val_end - val_start)))

        lr_scheduler.step()

        score_dic, cls_iu = metrics.get_scores()
        metrics.reset()
        val_loss_meter.reset()

        if score_dic['Mean IoU'] >= best_iou:
            best_iou = score_dic['Mean IoU']
            torch.save(model.state_dict(), f'./save_model/best_model.pth')
            # torch.save(model.state_dict(), f'./save_model/best_model_fake.pth')

        elif epoch % 20 == 0:
            torch.save(model.state_dict(), f'./save_model/{epoch}_model.pth')
            # torch.save((model.state_dict(), f'./save_model/{epoch}_model_fake.pth'))

        elif epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'./save_model/last_model.pth')
            # torch.save((model.state_dict(), f'./save_model/last_model_fake.pth'))

        epoch_end = time()

        print('Epoch: {} | Best MIoU: {} | Total Time: {:.4f}'.format(epoch + 1, best_iou, epoch_end - epoch_start))

    plt.plot(epoch_l, train_loss_l, 'ro--', label='train')
    plt.plot(epoch_l, val_loss_l, 'bo--', label='validation')
    plt.title('Real data')
    # plt.title('Fake data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'./fig/pretrained_loss.png')
    # plt.savefig(f'./fig/fake_loss.png')
    plt.close()


if __name__ == '__main__':
    main()
