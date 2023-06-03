import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

from btcv import BTCVDataSet
from unet3D import UNet3D
from loss_functions.score import *


def evaluate(model, test_data_loader, num_class, device):
    path = os.path.join('./fig', 'prediction_map')
    os.makedirs(path, exist_ok=True)

    metrics = Score(num_class)

    with torch.no_grad():
        model.eval()
        for valid_iter, pack in enumerate(test_data_loader):
            img = pack[0].to(device)
            label = pack[1].to(device)
            name = pack[2][0]

            pred = model(img)

            pred = torch.argmax(pred, dim=1).cpu().numpy()
            gt = label.data.cpu().numpy()
            metrics.update(gt, pred)

            # visualization
            # pred = np.copy(pred)[0, :, :]  # pred: 각 픽셀의 class
            # pred = pred.astype(int)
            # plt.figure()
            # plt.imshow(pred)
            # plt.title('Prediction Map')
            #
            # plt.savefig(f'{path}/{name}.png')
            # plt.close()

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
    test_data = BTCVDataSet(root=data_path, Train=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)

    # model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet3D(num_classes=n_classes)
    model.to(device)
    model.load_state_dict(torch.load(f'./save_model/best_model.pth'))

    # evaluation
    miou = evaluate(model, test_loader, n_classes, device=device)
    print(f'MIoU Score: {miou}')
