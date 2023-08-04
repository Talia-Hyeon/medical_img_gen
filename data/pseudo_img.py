import matplotlib.pyplot as plt
import torch

import sys

sys.path.append('..')
from test_unet import decode_segmap, find_best_view
from data.flare21 import *


class FAKEDataSet(data.Dataset):
    def __init__(self, root):
        self.root = root
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

            all_files.append({
                "image": img_file,
                "label": label_file,
                "name": item
            })

        self.files = all_files
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

        # normalization
        image = truncate(image)

        # add channel
        image = image[np.newaxis, :]

        # 50% flip
        # if np.random.rand(1) <= 0.5:  # W
        #     image = image[:, :, :, ::-1]
        #     label = label[:, :, :, ::-1]
        # if np.random.rand(1) <= 0.5:  # H
        #     image = image[:, :, ::-1, :]
        #     label = label[:, :, ::-1, :]
        # if np.random.rand(1) <= 0.5:  # D
        #     image = image[:, ::-1, :, :]
        #     label = label[:, ::-1, :, :]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label, name


def visualization(img, label, root, iter, num_classes):
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
    plt.title('Ground Truth')
    plt.savefig(f'{root}/{iter}.png')

    plt.close()


if __name__ == '__main__':
    num_classes = 5
    save_path = f'../fig/gen_img/mask'
    os.makedirs(save_path, exist_ok=True)
    val_path = f'../sample/mask'
    val_data = FAKEDataSet(root=val_path)
    val_loader = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)
    for val_iter, pack in enumerate(val_loader):
        img_ = pack[0]
        label_ = pack[1]
        visualization(img_, label_, save_path, val_iter, num_classes)
