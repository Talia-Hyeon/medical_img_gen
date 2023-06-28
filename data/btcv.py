import torch
from sklearn.model_selection import train_test_split

from data.flare21 import *

# 0: background
BTCV_label_num = {
    'spleen': 1,
    'right kidney': 2,
    'left kidney': 3,
    'gallbladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'inferior vena cava*': 9,
    'portal vein and splenic vein*': 10,
    'pancreas': 11,
    'right adrenal gland': 12,
    'left adrenal gland': 13
}

global valid_dataset
valid_dataset = {
    # 'background': 0,
    'liver': 1,
    'kidney': 2,
    'spleen': 3,
    'pancreas': 4,
}


class BTCVDataSet(data.Dataset):
    def __init__(self, root, split='train', task_id=1, crop_size=(192, 192, 192),
                 mean=(128, 128, 128), ignore_label=255):
        # crop_size=(64, 192, 192)
        self.root = root
        self.split = split
        self.task_id = task_id
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.ignore_label = ignore_label
        self.void_classes = [4, 5, 7, 8, 9, 10, 12, 13]

        print("Start preprocessing....")
        # load data
        image_path = osp.join(self.root, 'img')
        label_path = osp.join(self.root, 'label')
        img_list = os.listdir(image_path)

        # split train/val set
        train_data, rest_data = train_test_split(img_list, test_size=0.20, shuffle=True, random_state=0)
        val_data, test_data = train_test_split(rest_data, test_size=0.50, shuffle=True, random_state=0)

        if self.split == 'train':
            all_files = self.load_data(train_data, image_path, label_path)
            self.files = all_files

        elif self.split == 'val':
            all_files = self.load_data(val_data, image_path, label_path)
            self.files = all_files

        elif self.split == 'test':
            all_files = self.load_data(test_data, image_path, label_path)
            self.files = all_files

        print("{}'s {} images are loaded!".format(self.split, len(self.files)))

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

        # scale
        if self.split == 'train' and np.random.uniform() < 0.2:
            scaler = np.random.uniform(0.7, 1.4)
        else:
            scaler = 1

        # pad
        image = pad_image(image, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])
        label = pad_image(label, [self.crop_h * scaler, self.crop_w * scaler, self.crop_d * scaler])

        # crop
        [h0, h1, w0, w1, d0, d1] = locate_bbx(self.crop_d, self.crop_h, self.crop_w,
                                              label, scaler, datafiles["bbx"])
        image = image[h0: h1, w0: w1, d0: d1]
        label = label[h0: h1, w0: w1, d0: d1]

        # normalization & redefine label
        image = truncate(image)  # -1 <= image <= 1
        label = self.id2trainId(label)

        # add channel
        image = image[np.newaxis, :]
        label = label[np.newaxis, :]

        # Channel x Depth x H x W
        image = image.transpose((0, 3, 1, 2))
        label = label.transpose((0, 3, 1, 2))

        # 50% flip
        if self.split == 'train':
            if np.random.rand(1) <= 0.5:  # W
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            if np.random.rand(1) <= 0.5:  # H
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            if np.random.rand(1) <= 0.5:  # D
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]

        # adjust shape
        d, h, w = image.shape[-3:]
        if scaler != 1 or d != self.crop_d or h != self.crop_h or w != self.crop_w:
            image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (1, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True,
                           preserve_range=True)

        # extend label's channel for val/test
        label = extend_channel_classes(label, self.task_id)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name, labelNII.affine

    def load_data(self, data_l, image_path, label_path):
        all_files = []
        for i, item in enumerate(data_l):
            img_file = osp.join(image_path, item)
            label_item = item.replace('img', 'label')
            label_file = osp.join(label_path, label_item)

            label = nib.load(label_file).get_fdata()
            boud_h, boud_w, boud_d = np.where(label >= 1)  # background 아닌

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item,
                "bbx": [boud_h, boud_w, boud_d]
            })
        return all_files

    def id2trainId(self, label):
        # void_class to background
        for void_class in self.void_classes:
            label[np.where(label == void_class)] = 0

        # left kidney to kidney(2)
        label[np.where(label == 3)] = 2
        # spleen to 3
        label[np.where(label == 1)] = 3
        # liver to 1
        label[np.where(label == 6)] = 1
        # pancreas to 4
        label[np.where(label == 11)] = 4
        return label


if __name__ == '__main__':
    btcv = BTCVDataSet(root='../dataset/BTCV/Trainset', task_id=4)
    img_, label_, name_, label_aff = btcv[0]
    print("img's shape: {}\nlabel's shape: {}".format(img_.shape, label_.shape))
    # btcv = BTCVDataSet(root='../dataset/FLARE21', task_id=4)
    # valid_loader = data.DataLoader(dataset=btcv, batch_size=1, shuffle=False, num_workers=0)
    # for train_iter, pack in enumerate(valid_loader):
    #     img_ = pack[0]
    #     label_ = pack[1]
    #     name_ = pack[2]
    #     label_affine = pack[3]
    #     print("img_shape: {}\nlabel_shape: {}\naffine's type: {}".format(img_.shape, label_.shape, type(label_affine)))
