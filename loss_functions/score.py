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


# class DiceScore(nn.Module):
#     def __init__(self, num_classes=5):
#         super(DiceScore, self).__init__()
#         self.num_classes = num_classes
#         self.dice = BinaryDiceScore()
#
#     def forward(self, predict, target):
#         predict = F.softmax(predict, dim=1)
#
#         all_score = []
#         for i in range(1, self.num_classes):  # 1: evaluate score from organs(liver)
#             dice_score = self.dice(predict[:, i], target[:, i])
#             dice_score = torch.mean(dice_score)  # mean of batch
#             all_score.append(dice_score.item())  # append each organ
#
#         all_score = torch.tensor(all_score)
#         return all_score


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


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha  # teacher 모델의 출력을 얼마나 강력하게 사용할 지
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        # 1:channel 축소 & 0, targets.shape[1]: indexing background to # of prior channel
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


class MarginalLoss(nn.Module):
    def __init__(self, task_id=1):
        super(MarginalLoss, self).__init__()
        self.task_id = task_id
        self.criterion = BinaryDiceLoss()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        marg_pred = torch.zeros_like(target)
        for organ in range(self.task_id):
            marg_pred[:, 0] += predict[:, organ]
        marg_pred[:, 1] += predict[:, -1]

        total_loss = []
        for i in range(2):  # 0: background, 1: foreground
            dice_loss = self.criterion(marg_pred[:, i], target[:, i])
            total_loss.append(dice_loss)  # append each organ

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


class MarginalLossSupervised(nn.Module):
    def __init__(self, task_id=1, num_classes=5):
        super(MarginalLossSupervised, self).__init__()
        self.task_id = task_id
        self.num_classes = num_classes
        self.criterion = BinaryDiceLoss()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        marg_pred = torch.zeros_like(target)
        for organ in range(0, self.task_id):
            marg_pred[:, 0] += predict[:, organ]

        marg_pred[:, self.task_id] += predict[:, self.task_id]

        if self.task_id + 1 < self.num_classes:
            for organ in range(self.task_id + 1, self.num_classes):
                marg_pred[:, 0] += predict[:, organ]

        total_loss = []
        for i in range(self.num_classes):
            dice_loss = self.criterion(marg_pred[:, i], target[:, i])
            total_loss.append(dice_loss)  # append each organ

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


class CELoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super(CELoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = nn.BCELoss()

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
