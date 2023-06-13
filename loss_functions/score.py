import numpy as np
import torch
import torch.nn as nn


class Score:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - mean IU
        """
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {"Mean IoU": mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


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
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1, 2, 3])

                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]
