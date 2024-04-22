import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.flare21 import FLAREDataSet
from test_unet import find_best_view, decode_segmap
from util.AugGPU import Compose
from gen_img_class_vector import image_size


def visualization_augmentation(image, label, image_ag, label_ag, save_dir, name, num_classes):
    # remove batch
    image = torch.squeeze(image)
    image_ag = torch.squeeze(image_ag)

    # change binary to multi-class
    gt = torch.argmax(label, dim=0)
    gt_ag = torch.argmax(label_ag, dim=0)

    # save shape
    image_shape = list(image.shape)
    gt_shape = list(gt.shape)
    image_ag_shape = list(image.shape)
    gt_ag_shape = list(gt.shape)

    #  move to cpu & transform to numpy
    image = image.cpu().numpy()
    gt = gt.cpu().numpy()
    image_ag = image_ag.cpu().numpy()
    gt_ag = gt_ag.cpu().numpy()

    # slice into the best view
    max_score_idx = find_best_view(gt, num_classes)
    image = image[max_score_idx, :, :]
    gt = gt[max_score_idx, :, :]
    image_ag = image_ag[max_score_idx, :, :]
    gt_ag = gt_ag[max_score_idx, :, :]

    col_gt = decode_segmap(image, gt, num_classes)
    col_gt_ag = decode_segmap(image_ag, gt_ag, num_classes)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Image {image_shape}')
    plt.subplot(2, 2, 2)
    plt.imshow(col_gt)
    plt.title(f'GT {gt_shape}')
    plt.subplot(2, 2, 3)
    plt.imshow(image_ag, cmap='gray')
    plt.title(f'Image Augmentation {image_ag_shape}')
    plt.subplot(2, 2, 4)
    plt.imshow(col_gt_ag)
    plt.title(f'GT Augmentation {gt_ag_shape}')
    plt.savefig(f'{save_dir}/{name}.png')
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Visualization for Augmentation")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--save_dir", type=str, default='./fig/vis_aug/AF_EL_Flip')
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
    batch_size = args.batch_size
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # dataloader
    flared_path = './MOSInversion/dataset/FLARE_Dataset'
    flared_test = FLAREDataSet(root=flared_path, split='test')
    test_loader = DataLoader(dataset=flared_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    df = Compose(img_size=image_size, types=['F', 'E', 'A'],
                 grid_size=10, mag=6)

    for i, pack in enumerate(test_loader):
        imgs = pack[0]
        imgs = torch.tensor(imgs).to(device)
        labels = pack[1]
        labels = torch.tensor(labels).to(device)
        names = pack[2]

        imgs_ag, labels_ag = df(imgs, labels)

        for i in range(batch_size):
            visualization_augmentation(image=imgs[i], label=labels[i],
                                       image_ag=imgs_ag[i], label_ag=labels_ag[i],
                                       name=names[i], save_dir=save_dir, num_classes=5)
