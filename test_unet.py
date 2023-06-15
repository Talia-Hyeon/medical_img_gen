import os

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from data.btcv import BTCVDataSet
from model.unet3D import UNet3D
from loss_functions.score import *


def decode_segmap(temp):
    colors = [[0, 0, 0],  # "unlabelled"
              [220, 20, 60],
              [0, 0, 142],
              [0, 150, 20],
              [0, 0, 230]]
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


def evaluate(model, test_data_loader, num_class, device):
    path = os.path.join('./fig', 'prediction_map')
    # path = os.path.join('./fig', 'prediction_map_fake')
    os.makedirs(path, exist_ok=True)

    metrics = Score(num_class)

    with torch.no_grad():
        model.eval()
        for valid_iter, pack in enumerate(test_data_loader):
            img = pack[0]
            label = pack[1]
            name = pack[2][0]

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

    score_dic, cls_iu = metrics.get_scores()
    for k, v in score_dic.items():
        print(k, v)

    for k, v in cls_iu.items():
        print(k, v)

    metrics.reset()

    return score_dic['Mean IoU']


if __name__ == '__main__':
    n_classes = 4

    # dataloader
    data_path = './dataset/BTCV/Trainset'
    test_data = BTCVDataSet(root=data_path, task_id=4)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)

    # model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet3D(num_classes=n_classes)
    model.to(device)
    # model.load_state_dict(torch.load(f'./save_model/best_model.pth'))
    model.load_state_dict(torch.load(f'./save_model/best_model_fake.pth'))

    # evaluation
    miou = evaluate(model, test_loader, n_classes, device=device)
    print(f'MIoU Score: {miou}')
