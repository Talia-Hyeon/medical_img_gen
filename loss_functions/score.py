import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=4):
        super(CELoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])

                if self.weight is not None:
                    assert len(self.weight) == self.num_classes, \
                        'do not match length of weight and # of classes'
                    ce_loss *= self.weight[i]

                total_loss.append(ce_loss)  # append each organ

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of organ
        return avg_loss


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
    def __init__(self, weight=None, ignore_index=None, num_classes=4, **kwargs):
        super(DiceScore, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceScore(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):
        all_score = []
        if is_sigmoid:
            predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_score = self.dice(predict[:, i], target[:, i])

                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_score *= self.weights[i]

                dice_score = torch.mean(dice_score)  # mean of batch
                all_score.append(dice_score.item())  # append each organ

        all_score = torch.tensor(all_score)
        return all_score


class ArgmaxDiceScore(nn.Module):
    def __init__(self, ignore_index=None, num_classes=4, device=None, **kwargs):
        super(ArgmaxDiceScore, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device
        self.dice = BinaryDiceScore(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):
        total_socre = []
        if is_sigmoid:
            predict = F.sigmoid(predict)

        # add background channel
        shape = target[:, 0].shape
        backg = torch.zeros(shape).to(self.device)
        backg = backg.unsqueeze(1)
        predict = torch.cat((backg, predict), dim=1)
        target = torch.cat((backg, target), dim=1)

        # apply threshold 0.1
        predict = torch.threshold(predict, 0.1, 0)

        # one channel & multi-class
        predict = torch.argmax(predict, dim=1)
        target = torch.argmax(target, dim=1)

        # mutil-channel & binary class
        predict = self.extend_channel_classes(predict)
        target = self.extend_channel_classes(target)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_score = self.dice(predict[:, i], target[:, i])
                dice_score = torch.mean(dice_score)  # mean of each batch
                total_socre.append(dice_score.item())

        total_loss = torch.tensor(total_socre)
        return total_loss

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
    def __init__(self, weight=None, ignore_index=None, num_classes=4, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):

        total_loss = []
        if is_sigmoid:
            predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_loss *= self.weights[i]

                dice_loss = torch.mean(dice_loss)
                total_loss.append(dice_loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss
