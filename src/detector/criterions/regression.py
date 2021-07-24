import torch.nn.functional as F


def reg_loss(regr, gt_regr, mask):
    """L1 regression loss

    Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    """
    num = mask.float().sum()
    # print(gt_regr.size())
    mask = mask.sum(1).unsqueeze(1).expand_as(gt_regr)
    # print(mask.size())

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = F.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
