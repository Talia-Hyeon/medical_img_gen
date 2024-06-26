import torch
import torch.nn as nn
import torch.nn.functional as F


def extend_channel_classes(label, num_classes):
    label_list = []
    bg = torch.ones_like(label)

    for i in range(1, num_classes):
        label_i = torch.clone(label)
        label_i[label == i] = 1
        label_i[label != i] = 0
        bg -= label_i
        label_list.append(label_i)

    label_list = [bg] + label_list
    concated_label = torch.cat(label_list, dim=1)
    return concated_label


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


class sensitivity(nn.Module):
    # 민감도
    def __init__(self, smooth=1e-5):
        super(sensitivity, self).__init__()
        self.smooth = smooth

    def forward(self, predict, gt):
        TP = torch.sum(predict * gt)
        FN = torch.sum((1 - predict) * gt)

        sensitivity = TP / (TP + FN + self.smooth)
        return sensitivity


class specificity(nn.Module):
    # 특이도
    def __init__(self, smooth=1e-5):
        super(specificity, self).__init__()
        self.smooth = smooth

    def forward(self, predict, gt):
        TN = torch.sum((1 - predict) * (1 - gt))
        FP = torch.sum(predict * (1 - gt))

        specificity = TN / (TN + FP + self.smooth)
        return specificity


class ArgmaxScore(nn.Module):
    def __init__(self, mode='sensitivity', num_classes=5):
        super(ArgmaxScore, self).__init__()
        self.num_classes = num_classes
        if mode == 'sensitivity':
            self.criterion = sensitivity()
        elif mode == 'specificity':
            self.criterion = specificity()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)

        all_score = []
        # one channel & multi-class
        predict = torch.argmax(predict, dim=1, keepdim=True)  # (B, 1, H, W, D)
        # mutil-channel & binary class
        predict = extend_channel_classes(predict, num_classes=self.num_classes)  # (B, num_classes, H, W, D)
        for i in range(1, self.num_classes):  # 1: evaluate score from organs(liver)
            score = self.criterion(predict[:, i], target[:, i]).view(1, 1)  # (1)
            all_score.append(score)  # append each organ

        total_score = torch.cat(all_score, dim=1)  # (B, num_classes-1)
        return total_score



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
        predict = extend_channel_classes(predict, num_classes=self.num_classes)  # (B, num_classes, H, W, D)
        for i in range(1, self.num_classes):  # 1: evaluate score from organs(liver)
            dice_score = self.dice(predict[:, i], target[:, i]).unsqueeze(1)  # (B, 1)
            all_score.append(dice_score)  # append each organ

        total_score = torch.cat(all_score, dim=1)  # (B, num_classes-1)
        return total_score


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
        # predict = torch.sigmoid(predict)

        total_loss = []
        for i in range(self.num_classes):
            dice_loss = self.dice(predict[:, i], target[:, i])

            if self.weight is not None:
                assert len(self.weight) == self.num_classes, \
                    'do not match length of weight and # of classes'
                dice_loss *= self.weights[i]

            dice_loss = torch.mean(dice_loss)  # mean of batch
            total_loss.append(dice_loss)  # append each organ

        total_loss = torch.stack(total_loss)
        avg_loss = torch.mean(total_loss)  # mean of all organs
        return avg_loss


class MaskedLoss(nn.Module):
    def __init__(self, num_classes=5):
        super(MaskedLoss, self).__init__()
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss()
        # self.ce = CrossEntropyFnc()

    def forward(self, predict, target, task_id):
        # one channel & multi-class
        predict = torch.argmax(predict, dim=1, keepdim=True)  # (B, 1, H, W, D)
        # mutil-channel & binary class
        predict = extend_channel_classes(predict, num_classes=self.num_classes)  # (B, num_classes, H, W, D)

        loss_l = []
        for batch_id in range(len(task_id)):
            batch_task_id = task_id[batch_id]
            batch_pred = predict[batch_id, batch_task_id].unsqueeze(dim=0)
            batch_target = target[batch_id]
            dice_loss = self.dice(batch_pred, batch_target)
            # ce_loss = self.ce(batch_pred, batch_target)
            # batch_loss = dice_loss + ce_loss
            loss_l.append(dice_loss)
        loss = sum(loss_l)
        return loss


class CrossEntropyFnc(nn.Module):
    def __init__(self, smooth=1e-5):
        super(CrossEntropyFnc, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, label):
        # flatten
        prediction = prediction.contiguous().view(-1)
        label = label.contiguous().view(-1)

        loss = -torch.sum(label * torch.log(prediction + self.smooth))
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = CrossEntropyFnc()

    def forward(self, prediction, label):
        prediction = F.softmax(prediction, dim=1)
        loss = self.criterion(prediction, label)
        return loss
