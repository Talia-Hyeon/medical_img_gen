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

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_score


class ArgmaxDiceScore(nn.Module):
    def __init__(self, num_classes=5):
        super(ArgmaxDiceScore, self).__init__()
        self.num_classes = num_classes
        self.dice = BinaryDiceScore()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        all_score = []
        # one channel & multi-class
        predict = torch.argmax(predict, dim=1, keepdim=True)  # (B, 1, H, W, D)
        # mutil-channel & binary class
        predict = self.extend_channel_classes(predict)  # (B, num_classes, H, W, D)
        for i in range(1, self.num_classes):  # 1: evaluate score from organs(liver)
            dice_score = self.dice(predict[:, i], target[:, i]).unsqueeze(1)  # (B, 1)
            all_score.append(dice_score)  # append each organ

        total_score = torch.cat(all_score, dim=1)  # (B, num_classes-1)
        return total_score

    def extend_channel_classes(self, label):
        label_list = []
        bg = torch.ones_like(label)

        for i in range(1, self.num_classes):
            label_i = torch.clone(label)
            label_i[label == i] = 1
            label_i[label != i] = 0
            bg -= label_i
            label_list.append(label_i)

        label_list = [bg] + label_list
        concated_label = torch.cat(label_list, dim=1)
        return concated_label


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

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
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
            total_loss.append(dice_loss)  # append each organ

        # msg = 'DiceLoss '
        # for k, v in enumerate(total_loss):
        #     msg += f'| {index_organs[k]}: {v.item()} '
        # print(msg, end='\r')

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


class MarginalLoss(nn.Module):
    def __init__(self, task_id=1, num_classes=5):
        super(MarginalLoss, self).__init__()
        self.task_id = task_id
        self.num_classes = num_classes
        self.criterion = BinaryDiceLoss()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        # marg_pred = torch.zeros_like(target)
        # for organ in range(0, self.task_id):
        #     marg_pred[:, 0] += predict[:, organ]
        #
        # marg_pred[:, 1] += predict[:, self.task_id]
        #
        # if self.task_id + 1 < self.num_classes:
        #     for organ in range(self.task_id + 1, self.num_classes):
        #         marg_pred[:, 0] += predict[:, organ]

        marg_pred = torch.ones_like(target)
        marg_pred[:, 0] -= predict[:, self.task_id]
        marg_pred[:, 1] = predict[:, self.task_id]

        total_loss = []
        for i in range(2):  # backgroun, foreground
            dice_loss = self.criterion(marg_pred[:, i], target[:, i])
            total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of bg/fg
        return avg_loss


class SupervisedLoss(nn.Module):
    def __init__(self, num_classes=5):
        super(SupervisedLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, con_predict, con_target):
        binary_loss_l = []
        for task_id in range(1, self.num_classes):
            binary_predict = con_predict[task_id - 1].unsqueeze(dim=0)
            binary_target = con_target[task_id - 1].unsqueeze(dim=0)

            binary_criterion = MarginalLoss(task_id=task_id, num_classes=self.num_classes)
            binary_loss = binary_criterion(binary_predict, binary_target)
            binary_loss_l.append(binary_loss)

        loss = sum(binary_loss_l)
        return loss


class CELoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super(CELoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = cross_entropy_loss

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

        # msg = 'CELoss '
        # for k, v in enumerate(total_loss):
        #     msg += f'| {index_organs[k]}: {v.item()} '
        # print(msg, end='\r')

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


def cross_entropy_loss(prediction, label):
    # prediction = F.softmax(prediction, dim=1)

    # flatten
    prediction = prediction.view(-1)
    label = label.view(-1)

    # loss = -torch.sum(label * torch.log(prediction))
    loss = -torch.sum(label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction))
    return loss
