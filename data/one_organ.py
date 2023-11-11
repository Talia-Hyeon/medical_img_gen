import sys

import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

sys.path.append('..')
from data.flare21 import *
from test_unet import find_best_view, decode_segmap


class BinaryDataSet(data.Dataset):
    def __init__(self, task_id=1, crop_size=(160, 192, 192)):
        self.task_id = task_id
        self.crop_d, self.crop_h, self.crop_w = crop_size

        if self.task_id == 1:
            self.root = '../MOSInversion/dataset/Decathlon/Task03_Liver'
        elif self.task_id == 2:
            self.root = '../MOSInversion/dataset/KITS/data'
        elif self.task_id == 3:
            self.root = '../MOSInversion/dataset/Decathlon/Task09_Spleen'
        elif self.task_id == 4:
            self.root = '../MOSInversion/dataset/Decathlon/Task07_Pancreas'

        print("Start preprocessing....")
        # load data
        self.files = self.load_data()
        print("{} images are loaded!".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])
        image = imageNII.get_fdata()
        label = labelNII.get_fdata()
        name = datafiles["name"]

        # padding & crop
        if self.task_id != 2:
            image = pad_image(image, [self.crop_h, self.crop_w, self.crop_d])
            label = pad_image(label, [self.crop_h, self.crop_w, self.crop_d])
            image, label = center_crop_3d(image, label, self.crop_h, self.crop_w, self.crop_d)
            # Depth x H x W
            image = image.transpose((2, 0, 1))
            label = label.transpose((2, 0, 1))

        else:
            image = pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
            label = pad_image(label, [self.crop_d, self.crop_h, self.crop_w])
            image, label = center_crop_3d(image, label, self.crop_d, self.crop_h, self.crop_w)

        # normalization
        image = truncate(image)  # -1 <= image <= 1

        # add channel
        image = image[np.newaxis, :]
        label = label[np.newaxis, :]

        # 50% flip
        if np.random.rand(1) <= 0.5:  # W
            image = image[:, :, :, ::-1]
            label = label[:, :, :, ::-1]
        if np.random.rand(1) <= 0.5:  # H
            image = image[:, :, ::-1, :]
            label = label[:, :, ::-1, :]
        if np.random.rand(1) <= 0.5:  # D
            image = image[:, ::-1, :, :]
            label = label[:, ::-1, :, :]

        label = self.add_background(label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name

    def load_data(self):
        all_files = []
        if self.task_id == 2:
            path_list = os.listdir(self.root)

            for i, item in enumerate(path_list):
                img_file = osp.join(self.root, item, 'imaging.nii.gz')
                label_file = osp.join(self.root, item, 'segmentation.nii.gz')

                all_files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": item
                })

        else:
            image_path = osp.join(self.root, 'imagesTr')
            label_path = osp.join(self.root, 'labelsTr')
            all_img_list = os.listdir(image_path)
            img_list = [file for file in all_img_list if not file.startswith('.')]

            for i, item in enumerate(img_list):
                img_file = osp.join(image_path, item)
                label_file = osp.join(label_path, item)

                all_files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": item
                })

        return all_files

    def add_background(self, label):
        bg = np.ones_like(label)
        bg -= label
        stacked_label = np.concatenate([bg, label], axis=0)
        return stacked_label


if __name__ == '__main__':
    task_id = 4
    flare_path = '../dataset/FLARE21'
    flare_train = FLAREDataSet(root=flare_path, split='train', task_id=task_id)
    one_organ_data = BinaryDataSet(task_id=task_id)
    concat_data = ConcatDataset([flare_train, one_organ_data])
    print(len(concat_data))

    # num_classes = 2
    # path = '../fig/one_organ'
    # os.makedirs(path, exist_ok=True)
    # one_organ = BinaryDataSet(task_id=1)
    # train_loader = data.DataLoader(dataset=one_organ, batch_size=1, shuffle=False, num_workers=4)
    # for train_iter, pack in enumerate(train_loader):
    #     img = pack[0]
    #     label = pack[1]
    #     name = pack[2][0]
    #
    #     # remove batch
    #     img = torch.squeeze(img)
    #     label = torch.squeeze(label)
    #
    #     #  transform to numpy
    #     img = img.numpy()
    #     label = label.numpy()
    #
    #     # slice into the best view
    #     max_score_idx = find_best_view(label, num_classes)
    #     img = img[max_score_idx, :, :]
    #     label = label[max_score_idx, :, :]
    #     col_gt = decode_segmap(img, label, num_classes)
    #
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.title('Image')
    #     plt.imshow(img, cmap='gray')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(col_gt)
    #     plt.title('Ground Truth')
    #     plt.savefig(f'{path}/{name}.png')
    #     plt.close()
