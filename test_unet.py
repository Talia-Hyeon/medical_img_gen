import os
import argparse

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from data.flare21 import FLAREDataSet, index_organs
from model.unet3D import UNet3D
from loss_functions.score import *
from util import find_best_view


def decode_segmap(base, temp, num_classes):
    # m = np.mean(base)
    # s = np.std(base)
    # base = ((base - m) / np.sqrt(0.2 * s) + 0.5).clip(0.0, 1.0)
    # base = (base + 1) / 2.0  # 0-1 float
    base = base * 255  # 0-255
    # base = np.array(base, dtype='u1')  # uint8

    colors = [[0, 0, 0],  # "unlabelled"
              [150, 0, 0],
              [0, 0, 142],
              [150, 170, 0],
              [70, 0, 100]]
    label_colours = dict(zip(range(num_classes), colors))  # key: label's index, value: color

    r = base.copy()
    g = base.copy()
    b = base.copy()
    for l in range(1, num_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # print("rgb's dtype: {}".format(rgb.dtype))
    return rgb


def visualization(img, label, pred, name, path, num_classes):
    # remove batch
    img = torch.squeeze(img)
    pred = torch.squeeze(pred)
    label = torch.squeeze(label)

    # change binary to multi-class
    pred = torch.argmax(pred, dim=0)
    gt = torch.argmax(label, dim=0)

    #  move to cpu & transform to numpy
    img = img.cpu().numpy()
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    # slice into the best view
    max_score_idx = find_best_view(gt, num_classes)
    img = img[max_score_idx, :, :]
    pred = pred[max_score_idx, :, :]
    gt = gt[max_score_idx, :, :]

    col_pred = decode_segmap(img, pred, num_classes)
    col_gt = decode_segmap(img, gt, num_classes)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(col_pred)
    plt.title('Prediction Map')
    plt.subplot(1, 2, 2)
    plt.imshow(col_gt)
    plt.title('Ground Truth')
    plt.savefig(f'{path}/{name}.png')
    plt.close()


def evaluate(model, test_data_loader, num_class, device, train_type):
    path = f'./fig/prediction_map/{train_type}'
    os.makedirs(path, exist_ok=True)

    metric = ArgmaxDiceScore(num_classes=num_class)
    dice_list = []

    with torch.no_grad():
        model.eval()
        for valid_iter, pack in enumerate(test_data_loader):
            img = pack[0]
            label = pack[1]
            name = pack[2][0]

            img = img.to(device)
            label = label.to(device)
            pred = model(img)

            iter_dice = metric(pred, label)
            dice_list.append(iter_dice)

            visualization(img, label, pred, name, path, num_class)

    total_dice = torch.cat(dice_list, dim=0)
    dice_score = torch.mean(total_dice, dim=0)
    return dice_score


def print_dice(dice_score):
    label_dict = {}
    for idx, organ in enumerate(index_organs[1:]):
        label_dict[idx] = organ
    dice_dict = {}
    for i in range(len(label_dict)):
        organ = label_dict[i]
        dice_dict[organ] = dice_score[i].item()

    print(dice_dict)
    avg_dice = torch.mean(dice_score).item()
    print('Average_Dice_Score: {}'.format(avg_dice))
    return dice_dict


def get_args():
    parser = argparse.ArgumentParser(description="test_pretrained_UNet")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--type", type=str, default='flare')
    parser.add_argument("--model_path", type=str, default='./save_model/best_model.pth')
    return parser


if __name__ == '__main__':
    parser = get_args()
    print(parser)
    args = parser.parse_args()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameter
    train_type = args.type
    n_classes = args.num_classes
    num_workers = args.num_workers

    # dataloader
    flared_path = './dataset/FLARE_Dataset'
    flared_test = FLAREDataSet(root=flared_path, split='test')
    test_loader = DataLoader(dataset=flared_test, batch_size=1, shuffle=False, num_workers=num_workers)

    # model
    model = UNet3D(num_classes=n_classes)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = nn.DataParallel(model).to(device)

    # evaluation
    dice = evaluate(model, test_loader, n_classes, device=device, train_type=train_type)
    print_dice(dice)
