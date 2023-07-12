import os
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset
from matplotlib import pyplot as plt

from data.btcv import *
from data.flare21 import FLAREDataSet
from model.unet3D import UNet3D
from loss_functions.score import *


def decode_segmap(temp):
    colors = [[0, 0, 0],  # "unlabelled"
              [150, 0, 0],
              [0, 0, 142],
              [150, 170, 0],
              [70, 0, 100]]
    label_colours = dict(zip(range(5), colors))  # key: label's index, value: color

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(5):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def find_best_view(img):
    d1, d2, d3 = img.shape
    max_score = 0
    max_score_idx = 0
    for i in range(d1):
        sagital_pred = img[i, :, :]
        classes = np.unique(sagital_pred)
        if classes.size >= 2:
            counts = np.array([max(np.where(sagital_pred == c)[0].size, 1e-8) for c in range(5)])  # 5:num_classes
            score = np.exp(np.sum(np.log(counts)) - 5 * np.log(np.sum(counts)))
            if score > max_score:
                max_score = score
                max_score_idx = i
    return max_score_idx


def visualization(img, label, pred, name, path, device):
    # add background channel
    shape = label[:, 0].shape
    backg = torch.zeros(shape).to(device)
    backg = backg.unsqueeze(0)
    pred = torch.cat((backg, pred), dim=1)
    label = torch.cat((backg, label), dim=1)

    # apply threshold
    pred = torch.threshold(pred, 0.1, 0)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    gt = torch.argmax(label, dim=1).cpu().numpy()
    img = img.cpu().numpy()

    # remove batch, channel
    img = np.squeeze(img, axis=0)
    img = np.squeeze(img, axis=0)
    pred = np.squeeze(pred, axis=0)
    gt = np.squeeze(gt, axis=0)

    max_score_idx = find_best_view(gt)
    img = img[max_score_idx, :, :]
    pred = pred[max_score_idx, :, :]
    gt = gt[max_score_idx, :, :]

    col_pred = decode_segmap(pred)
    col_gt = decode_segmap(gt)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(col_pred)
    plt.title('Prediction Map')
    plt.subplot(1, 3, 3)
    plt.imshow(col_gt)
    plt.title('Ground Truth')
    plt.savefig(f'{path}/{name}.png')
    plt.close()


def evaluate(model, test_data_loader, num_class, device):
    path = os.path.join('./fig', 'prediction_map')
    os.makedirs(path, exist_ok=True)

    metric = ArgmaxDiceScore(num_classes=num_class+1, device=device)
    # metric = DiceScore(num_classes=num_class)
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

            visualization(img, label, pred, name, path, device)

    total_dice = torch.stack(dice_list)
    dice_score = torch.mean(total_dice, dim=0) # mean of all batches
    return dice_score


def print_dice(dice_score):
    label_dict = {value: key for key, value in valid_dataset.items()}
    label_dict[0] = 'background'
    dice_dict = {}
    for i in range(len(label_dict)):
        organ = label_dict[i]  # [i+1]
        dice_dict[organ] = dice_score[i].item()

    print(dice_dict)
    avg_dice = torch.mean(dice_score).item()
    print('Average_Dice_Score: {}'.format(avg_dice))
    return dice_dict


def get_args():
    parser = argparse.ArgumentParser(description="test_pretrained_UNet")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--model_path", type=str, default='./save_model/199_last_model.pth')
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
    n_classes = args.num_classes
    num_workers = args.num_workers

    # dataloader
    flared_path = './dataset/FLARE21'
    # btcv_path = './dataset/BTCV/Trainset'
    flared_test = FLAREDataSet(root=flared_path, split='test', task_id=n_classes)
    # btcv_test = BTCVDataSet(root=btcv_path, split='test', task_id=n_classes)
    # test_data = ConcatDataset([flared_test, btcv_test])
    test_loader = DataLoader(dataset=flared_test, batch_size=1, shuffle=False, num_workers=num_workers)

    # model
    model = UNet3D(num_classes=n_classes)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = nn.DataParallel(model).to(device)

    # evaluation
    dice = evaluate(model, test_loader, n_classes, device=device)
    print_dice(dice)
