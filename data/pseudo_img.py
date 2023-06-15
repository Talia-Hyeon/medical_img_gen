import matplotlib.pyplot as plt

from test_unet import find_best_view, decode_segmap
from data.flare21 import *


class FAKEDataSet(data.Dataset):
    def __init__(self, root, task_id=1, crop_size=(64, 192, 192), mean=(128, 128, 128), ignore_label=255,
                 split='train'):
        self.root = root
        self.task_id = task_id
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.ignore_label = ignore_label
        self.split = split
        self.files = []

        print("Start preprocessing....")
        # load data
        image_path = osp.join(self.root, 'Img')
        label_path = osp.join(self.root, 'Pred')

        img_list = os.listdir(image_path)
        all_files = []
        for i, item in enumerate(img_list):
            img_file = osp.join(image_path, item)
            label_item = item.replace('img', 'pred')
            label_file = osp.join(label_path, label_item)

            label = nib.load(label_file).get_fdata()
            boud_h, boud_w, boud_d = np.where(label >= 1)  # background 아닌

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item,
                "bbx": [boud_h, boud_w, boud_d]
            })

        # split train/val set
        train_X, val_X = train_test_split(all_files, test_size=0.20, shuffle=True, random_state=0)
        if self.split == 'train':
            self.files = train_X
        else:
            self.files = val_X

        print('{} images are loaded!'.format(len(self.files)))

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

        # normalization
        image = truncate(image)

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

        # extend label's channel to # of classes for loss fn
        label = extend_channel_classes(label, self.task_id)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name


def visualization(img, label, root, iter):
    image = img.numpy()
    image = np.squeeze(image)
    label = label.numpy()
    label = np.squeeze(label)
    label = np.argmax(label, axis=0)

    max_score_idx = find_best_view(label)
    image = image[max_score_idx, :, :]
    label = label[max_score_idx, :, :]
    col_label = decode_segmap(label)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(col_label)
    plt.title('Ground Truth')
    plt.savefig(f'{root}/{iter}.png')

    plt.close()


if __name__ == '__main__':
    save_path = '../fig/gen_img'
    os.makedirs(save_path, exist_ok=True)
    val_path = '../sample'
    val_data = FAKEDataSet(root=val_path, split='val', task_id=4)
    val_loader = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)
    for val_iter, pack in enumerate(val_loader):
        img_ = pack[0]
        label_ = pack[1]
        visualization(img_, label_, save_path, val_iter)
