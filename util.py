import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose

from model.unet3D import UNet3D


def save_model(path, model, optim, epoch):
    net_state = model.module.state_dict()
    states = {
        'model': net_state,
        'optimizer': optim.state_dict(),
        # 'scheduler': lr_sch.state_dict(),
        'epoch': epoch + 1
    }
    torch.save(states, path)


def load_model(num_classes=5, check_point=False):
    pre_path = f'./save_model/last_model.pth'
    if os.path.exists(pre_path) != True:
        dirname, basename = os.path.split(pre_path)
        pre_path = dirname + '/best_model.pth'

    pretrained = UNet3D(num_classes=num_classes)
    checkpoint = torch.load(pre_path)
    pretrained.load_state_dict(checkpoint['model'], strict=False)

    if check_point == True:
        return pretrained, checkpoint
    else:
        return pretrained


def visualization(img, label, root, name, num_classes):
    # delete batch
    image = torch.squeeze(img).numpy()
    label = torch.squeeze(label).numpy()
    label = np.argmax(label, axis=0)

    # slice into the best view
    max_score_idx = find_best_view(label, num_classes)
    image = image[max_score_idx, :, :]
    label = label[max_score_idx, :, :]
    col_label = decode_segmap(label, num_classes)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(col_label)
    plt.title('Prediction')
    plt.savefig(f'{root}/{name}.png')

    plt.close()

    # d, h, w = label.shape
    # for i in range(d):
    #     image_i = image[i, :, :]
    #     label_i = label[i, :, :]
    #     col_label = decode_segmap(label_i, num_classes)
    #
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image_i, cmap='gray')
    #     plt.title('Image')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(col_label)
    #     plt.title('Prediction')
    #     plt.savefig(f'{root}/{name}_{i}.png')
    #
    #     plt.close()


def find_best_view(img, num_classes):
    d1, d2, d3 = img.shape
    max_score = 0
    max_score_idx = 0
    for i in range(d1):
        sagital_pred = img[i, :, :]
        classes = np.unique(sagital_pred)
        if classes.size >= 2:
            counts = np.array([max(np.where(sagital_pred == c)[0].size, 1e-8) for c in range(num_classes)])
            score = np.exp(np.sum(np.log(counts)) - num_classes * np.log(np.sum(counts)))
            if score > max_score:
                max_score = score
                max_score_idx = i
    return max_score_idx


def decode_segmap(temp, num_classes):
    colors = [[0, 0, 0],  # "unlabelled"
              [150, 0, 0],
              [0, 0, 142],
              [150, 170, 0],
              [70, 0, 100]]
    label_colours = dict(zip(range(num_classes), colors))  # key: label's index, value: color

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(1, num_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True,
                              p_per_channel=0.5, p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15,
                                             p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True,
                                        retain_stats=True, p_per_sample=0.15, data_key="image"))

    # compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'image': image,
                 'label': label,
                 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


def task_collate(batch):
    image, label, name, task_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)
    data_dict = {'image': image,
                 'label': label,
                 'name': name,
                 'task_id': task_id}

    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    return data_dict


def truncate(CT):
    min_HU = -325
    max_HU = 325
    subtract = 0
    divide = 325.

    # truncate
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    CT = CT - subtract
    CT = CT / divide
    return CT
