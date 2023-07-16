import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.flare21 import index_organs

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BinaryDiceScore(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target), dim=1)
        union = torch.sum(predict, dim=1) + torch.sum(target, dim=1)

        dice_score = 2 * intersection / (union + self.smooth)
        return dice_score


class DiceScore(nn.Module):
    def __init__(self, num_classes=5):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.dice = BinaryDiceScore()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        all_score = []
        for i in range(1, self.num_classes):  # 1: evaluate score from organs(liver)
            dice_score = self.dice(predict[:, i], target[:, i])
            dice_score = torch.mean(dice_score)  # mean of batch
            all_score.append(dice_score.item())  # append each organ

        all_score = torch.tensor(all_score)
        return all_score


class ArgmaxDiceScore(nn.Module):
    def __init__(self, num_classes=5, device=None):
        super(ArgmaxDiceScore, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.dice = BinaryDiceScore()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        total_score = []
        # one channel & multi-class
        predict = torch.argmax(predict, dim=1)
        target = torch.argmax(target, dim=1)

        # mutil-channel & binary class
        predict = self.extend_channel_classes(predict)
        target = self.extend_channel_classes(target)

        for i in range(self.num_classes):
            dice_score = self.dice(predict[:, i], target[:, i])
            dice_score = torch.mean(dice_score)  # mean of each batch
            total_score.append(dice_score.item())

        total_score = torch.tensor(total_score)
        return total_score

    def extend_channel_classes(self, label):
        label_list = []
        for i in range(1, self.num_classes + 1):
            label_i = torch.clone(label)
            label_i[label == i] = 1
            label_i[label != i] = 0
            label_list.append(label_i)
        stacked_label = torch.stack(label_list, axis=1)
        stacked_label = torch.squeeze(stacked_label)
        return stacked_label


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target), dim=1)
        union = torch.sum(predict, dim=1) + torch.sum(target, dim=1)

        dice_score = 2 * intersection / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        total_loss = []
        for i in range(self.num_classes):
            dice_loss = self.dice(predict[:, i], target[:, i])

            if self.weight is not None:
                assert len(self.weight) == self.num_classes, \
                    'do not match length of weight and # of classes'
                dice_loss *= self.weights[i]

            dice_loss = torch.mean(dice_loss)  # mean of batch
            total_loss[i] = dice_loss  # append each organ

        msg = 'DiceLoss '
        for k, v in total_loss.items():
            msg += f'| {index_organs[k]}: {v.item()} '
        print(msg, end='\r')

        total_loss = torch.stack(total_loss.item())
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


class CELoss(nn.Module):
    def __init__(self, weight=None, num_classes=4):
        super(CELoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        predict = F.softmax(predict, dim=1)

        total_loss = []
        for i in range(self.num_classes):
            ce_loss = self.criterion(predict[:, i], target[:, i])

            if self.weight is not None:
                assert len(self.weight) == self.num_classes, \
                    'do not match length of weight and # of classes'
                ce_loss *= self.weight[i]

            total_loss.append(ce_loss)  # append each organ

        msg = 'CELoss '
        for k, v in total_loss.items():
            msg += f'| {index_organs[k]}: {v.item()} '
        print(msg, end='\r')

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss
