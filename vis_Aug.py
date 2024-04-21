import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.flare21 import FLAREDataSet
from test_unet import find_best_view, decode_segmap
from util.AugGPU import Random_EL, Random_AF
from gen_img_class_vector import image_size


def visualization_augmentation(image, label, save_dir, name, num_classes):
    # remove batch
    image = torch.squeeze(image)
    label = torch.squeeze(label)

    # change binary to multi-class
    gt = torch.argmax(label, dim=0)

    #  move to cpu & transform to numpy
    image = image.cpu().numpy()
    gt = gt.cpu().numpy()

    # slice into the best view
    max_score_idx = find_best_view(gt, num_classes)
    image = image[max_score_idx, :, :]
    gt = gt[max_score_idx, :, :]

    col_gt = decode_segmap(image, gt, num_classes)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(col_gt)
    plt.title('Ground Truth')
    plt.savefig(f'{save_dir}/{name}.png')
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Visualization for Augmentation")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default='0,1')
    parser.add_argument("--save_dir", type=str, default='./fig/vis_aug')
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
    flared_path = './dataset/FLARE_Dataset'
    flared_test = FLAREDataSet(root=flared_path, split='test', task_id=n_classes - 1)
    test_loader = DataLoader(dataset=flared_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    df = Random_EL(img_size=image_size, grid_size=10, mag=6)

    for i, pack in enumerate(test_loader):
        img = pack['image']
        img = torch.tensor(img).to(device)
        label = pack['label']
        label = torch.tensor(label).to(device)
        name = pack['name']

        img, label = df(img, label)

        visualization_augmentation(image=img, label=label, name=name, save_dir=save_dir, num_classes=5)
