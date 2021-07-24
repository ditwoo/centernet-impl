import warnings

import numpy as np
import torch


def change_box_order(boxes, order):
    """Change box order between
    (xmin, ymin, xmax, ymax) <-> (xcenter, ycenter, width, height).

    Args:
        boxes: (torch.Tensor or np.ndarray) bounding boxes, sized [N,4].
        order: (str) either "xyxy2xywh" or "xywh2xyxy".

    Returns:
        (torch.Tensor) converted bounding boxes, sized [N,4].
    """
    assert order in {"xyxy2xywh", "xywh2xyxy"}
    concat_fn = torch.cat if isinstance(boxes, torch.Tensor) else np.concatenate

    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return concat_fn([(a + b) / 2, b - a], 1)
    return concat_fn([a - b / 2, a + b / 2], 1)


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    """Clamp boxes.

    Args:
        boxes: (torch.Tensor or numpy.ndarray) bounding boxes with format
            (xmin, ymin, xmax, ymax) and shape [N, 4].
        xmin: (int or float) min value of x.
        ymin: (int or float) min value of y.
        xmax: (int or float) max value of x.
        ymax: (int or float) max value of y.

    Returns:
        clamped boxes (torch.Tensor or numpy.ndarray).
    """
    if isinstance(boxes, torch.Tensor):
        res = boxes.detach().clone()
        res[:, 0].clamp_(min=xmin, max=xmax)
        res[:, 1].clamp_(min=ymin, max=ymax)
        res[:, 2].clamp_(min=xmin, max=xmax)
        res[:, 3].clamp_(min=ymin, max=ymax)
    else:
        res = np.copy(boxes)
        res[:, 0] = np.clip(res[:, 0], a_min=xmin, a_max=xmax)
        res[:, 1] = np.clip(res[:, 1], a_min=ymin, a_max=ymax)
        res[:, 2] = np.clip(res[:, 2], a_min=xmin, a_max=xmax)
        res[:, 3] = np.clip(res[:, 3], a_min=ymin, a_max=ymax)
    return res


def box_select(boxes, xmin, ymin, xmax, ymax):
    """Select boxes in range (xmin, ymin, xmax, ymax).

    Args:
        boxes: (torch.Tensor or numpy.ndarray) bounding boxes with format
            (xmin, ymin, xmax, ymax) and shape [N, 4].
        xmin: (int or float) min value of x.
        ymin: (int or float) min value of y.
        xmax: (int or float) max value of x.
        ymax: (int or float) max value of y.

    Returns:
        selected boxes (torch.Tensor or numpy.ndarray) with shape [M,4].
        selected mask (torch.Tensor or numpy.ndarray) with shape [N,].
    """
    mask = (boxes[:, 0] >= xmin) & (boxes[:, 1] >= ymin) & (boxes[:, 2] <= xmax) & (boxes[:, 3] <= ymax)
    boxes = boxes[mask, :]
    return boxes, mask


def _numpu_box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (np.ndarray) bounding boxes with shape [N, 4].
        box2: (np.ndarray) bounding boxes with shape [M, 4].

    Return:
        iou (np.ndarray): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)  # [N, M]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iou = inter / (area1[:, None] + area2 - inter)
    return iou


def _torch_box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (torch.Tensor) bounding boxes with shape [N, 4].
        box2: (torch.Tensor) bounding boxes with shape [M, 4].

    Return:
        iou (torch.Tensor): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (torch.Tensor or np.ndarray) bounding boxes with shape [N,4].
        box2: (torch.Tensor or np.ndarray) bounding boxes with shape [M,4].

    Return:
        iou (torch.Tensor or np.ndarray): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor):
        return _torch_box_iou(box1, box2)
    else:
        return _numpu_box_iou(box1, box2)


def _torch_box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (torch.Tensor) bounding boxes with shape [N,4].
        scores: (torch.Tensor) confidence scores with shape [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (torch.Tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    if bboxes.dim() == 1:
        bboxes = bboxes.reshape(-1, 4)

    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long)


def _numpy_box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (np.ndarray) bounding boxes, sized [N,4].
        scores: (np.ndarray) confidence scores, sized [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (np.ndarray) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (torch.Tensor) bounding boxes with shape [N,4].
        scores: (torch.Tensor) confidence scores with shape [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (torch.Tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    if isinstance(bboxes, torch.Tensor) and isinstance(scores, torch.Tensor):
        return _torch_box_nms(bboxes, scores, threshold)
    else:
        return _numpy_box_nms(bboxes, scores, threshold)


def nms_filter(bboxes, classes, confidences, iou_threshold=0.5):
    """Filter classes, bboxes, confidences by nms with iou_threshold.

    Args:
        bboxes (np.ndarray): array with bounding boxes, expected shape [N, 4].
        classes (np.ndarray): array with classes, expected shape [N,].
        confidences (np.ndarray)): array with class confidence, expected shape [N,].
        iou_threshold (float): IoU threshold to use for filtering.
            Default is ``0.5``.

    Returns:
        filtered bboxes (np.ndarray), classes (np.ndarray), and confidences (np.ndarray)
            where number of records will be equal to some M (M <= N).
    """
    keep_bboxes = []
    keep_classes = []
    keep_confidences = []

    for presented_cls in np.unique(classes):
        mask = classes == presented_cls
        curr_bboxes = bboxes[mask, :]
        curr_classes = classes[mask]
        curr_confs = confidences[mask]

        to_keep = box_nms(curr_bboxes, curr_confs, iou_threshold)

        keep_bboxes.append(curr_bboxes[to_keep, :])
        keep_classes.append(curr_classes[to_keep])
        keep_confidences.append(curr_confs[to_keep])

    return np.concatenate(keep_bboxes), np.concatenate(keep_classes), np.concatenate(keep_confidences)
