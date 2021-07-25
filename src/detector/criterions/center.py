import torch
import torch.nn as nn

from .negative import neg_loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes=1, mask_loss_weight=1.0, regr_loss_weight=1.0, size_average=True):
        super().__init__()
        self.num_classes = num_classes
        self.mask_loss_weight = mask_loss_weight
        self.regr_loss_weight = regr_loss_weight
        self.size_average = size_average

    def forward(self, predicted_heatmap, predicted_regr, target_heatmap, target_regr):
        pred_mask = torch.sigmoid(predicted_heatmap)
        mask_loss = neg_loss(pred_mask, target_heatmap)
        mask_loss *= self.mask_loss_weight

        regr_loss = (
            torch.abs(predicted_regr - target_regr).sum(1)[:, None, :, :] * target_heatmap
        ).sum()  # .sum(1).sum(1).sum(1)
        regr_loss = regr_loss / target_heatmap.sum()  # .sum(1).sum(1).sum(1)
        regr_loss *= self.regr_loss_weight

        loss = mask_loss + regr_loss
        if not self.size_average:
            loss *= predicted_heatmap.shape[0]

        return loss, mask_loss, regr_loss


def center_loss(prediction, mask, regr, weight=0.4, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = neg_loss(pred_mask, mask)

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss
