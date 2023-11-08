import os
import torch
from model.unet3D import UNet3D


def save_model(path, model, optim, lr_sch, epoch):
    net_state = model.module.state_dict()
    states = {
        'model': net_state,
        'optimizer': optim.state_dict(),
        'scheduler': lr_sch.state_dict(),
        'epoch': epoch + 1
    }
    torch.save(states, path)


def load_model(train_type, task_id):
    pre_path = f'./save_model/{train_type}/{task_id - 1}/last_model.pth'
    if os.path.exists(pre_path) != True:
        dirname, basename = os.path.split(pre_path)
        pre_path = dirname + '/best_model.pth'
    pretrained = UNet3D(num_classes=task_id)
    checkpoint = torch.load(pre_path)
    pretrained.load_state_dict(checkpoint['model'], strict=False)
    return pretrained
