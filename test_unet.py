import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

from btcv import BTCVDataSet
from unet3D import UNet3D
from loss_functions.score import *


def evaluate(model, test_data_loader, num_class, device, crops=(100, 480, 480)):
    path = os.path.join('./fig', 'prediction_map')
    os.makedirs(path, exist_ok=True)

    metrics = Score(num_class)

    crop_d, crop_h, crop_w = crops

    with torch.no_grad():
        model.eval()
        for valid_iter, pack in enumerate(test_data_loader):
            img = pack[0]
            label = pack[1]
            name = pack[2][0]

            B, C, D, H, W = img.shape
            d0 = (D - crop_d) // 2
            d1 = (D + crop_d) // 2
            h0 = (H - crop_h) // 2
            h1 = (H + crop_d) // 2
            w0 = (W - crop_w) // 2
            w1 = (W + crop_w) // 2

            img = img[:, :, d0:d1, h0:h1, w0:w1]
            label = label[:, :, d0:d1, h0:h1, w0:w1]

            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            pred = torch.sigmoid(pred)

            img = img.cpu().numpy()
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            gt = torch.argmax(label, dim=1).cpu().numpy()
            metrics.update(gt, pred)

            # visualization
            img = np.squeeze(img, axis=0)
            img = np.squeeze(img, axis=0)
            pred = np.squeeze(pred, axis=0)
            gt = np.squeeze(gt, axis=0)
            d1, d2, d3 = pred.shape
            max_score = 0
            max_score_idx = 0
            for i in range(d3):
                sagital_pred = pred[:, :, i]
                classes = np.unique(sagital_pred)
                if classes.size >= 1:
                    counts = np.array([max(np.where(sagital_pred == c)[0].size, 1e-8) for c in range(num_class)])
                    score = np.exp(np.sum(np.log(counts)) - 5 * np.log(np.sum(counts)))
                    if score > max_score:
                        max_score = score
                        max_score_idx = i

            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, :, max_score_idx], cmap='gray')
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(pred[:, :, max_score_idx])
            plt.title('Prediction Map')
            plt.subplot(1, 3, 3)
            plt.imshow(gt[:, :, max_score_idx])
            plt.title('Ground Truth')
            plt.savefig(f'{path}/{name}.png')
            plt.close()

    score_dic, cls_iu = metrics.get_scores()
    for k, v in score_dic.items():
        print(k, v)

    for k, v in cls_iu.items():
        print(k, v)

    metrics.reset()

    return score_dic['Mean IoU']


if __name__ == '__main__':
    n_classes = 5

    # dataloader
    data_path = './dataset/BTCV/Trainset'
    test_data = BTCVDataSet(root=data_path, split='test')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)

    # model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet3D(num_classes=n_classes)
    model.to(device)
    model.load_state_dict(torch.load(f'./save_model/best_model.pth'))

    # evaluation
    miou = evaluate(model, test_loader, n_classes, device=device)
    print(f'MIoU Score: {miou}')
