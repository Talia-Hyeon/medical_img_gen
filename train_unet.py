import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from btcv import BTCVDataSet
from unet3D import UNet3D
from loss_functions.loss import *
from loss_functions.score import *


def main():
    os.makedirs('./save_model', exist_ok=True)
    os.makedirs('./fig', exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Hyper-parameters
    num_epochs = 100
    n_classes = 4

    model = UNet3D(num_classes=n_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=0.0001
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # data loader
    data_path = './dataset/BTCV/Trainset'
    train_data = BTCVDataSet(root=data_path, Train=True)
    valid_data = BTCVDataSet(root=data_path, Train=False)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=0)

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
        for train_iter, pack in enumerate(train_loader):
            img = pack[0].to(device)
            label = pack[1].to(device)
            pred = model(img)

            loss = CELoss4MOTS(pred, label)
            train_loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_iter + 1) % (len(train_loader) // 5) == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} | Iters: {train_iter + 1} | Train loss: {loss.item():.4f}')

        train_loss_l.append(train_loss_meter.avg)
        epoch_l.append(epoch)
        train_loss_meter.reset()

        with torch.no_grad():
            model.eval()
            for valid_iter, pack in enumerate(valid_loader):
                img_val = pack[0].to(device)
                label_val = pack[1].to(device)

                pred_val = model(img_val)
                val_loss = CELoss4MOTS(pred_val, label_val)

                pred_val = torch.argmax(pred, dim=1).cpu().numpy()
                gt = label_val.data.cpu().numpy()

                metrics.update(gt, pred_val)
                val_loss_meter.update(val_loss.item())

            val_loss_l.append(val_loss_meter.avg)
            print(f'Epoch: {epoch + 1} | Valid loss: {val_loss.item():.4f}')

        lr_scheduler.step()

        score_dic, cls_iu = metrics.get_scores()
        metrics.reset()
        val_loss_meter.reset()

        if score_dic['Mean IoU'] >= best_iou:
            best_iou = score_dic['Mean IoU']
            torch.save(model.state_dict(), f'./save_model/best_model.pth')
        print('Epoch: {} | Best MIoU: {}'.format(epoch + 1, best_iou))

    plt.plot(epoch_l, train_loss_l, 'ro--', label='train')
    plt.plot(epoch_l, val_loss_l, 'bo--', label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.savefig(f'./fig/loss.png')
    plt.close()


if __name__ == '__main__':
    main()
