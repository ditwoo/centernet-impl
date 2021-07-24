import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from detector.datasets.utils import load_coco_json, pixels_to_absolute, read_image

IN_SCALE = 1


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


def draw_msra_gaussian(heatmap, center, sigma=2):
    tmp_size = sigma * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = (max(0, -ul[0]), min(br[0], h) - ul[0])
    g_y = (max(0, -ul[1]), min(br[1], w) - ul[1])
    img_x = (max(0, ul[0]), min(br[0], h))
    img_y = (max(0, ul[1]), min(br[1], w))
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(  # noqa: E203
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],  # noqa: E203
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],  # noqa: E203
    )
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]  # noqa: E203
    masked_regmap = regmap[:, y - top : y + bottom, x - left : x + right]  # noqa: E203
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]  # noqa: E203
    masked_reg = reg[:, radius - top : radius + bottom, radius - left : radius + right]  # noqa: E203
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top : y + bottom, x - left : x + right] = masked_regmap  # noqa: E203
    return regmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]  # noqa: E203
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def pred2box(heatmap, regr, threshold=0.5, scale=4, input_size=512):
    """Convert model output to bounding boxes.

    Args:
        heatmap (np.ndarray): center heatmap, expected matrix with shapes [N, N].
        regr (np.ndarray): width and height coordinates, expected matrix with shapes [2, N, N]
        threshold (float): score threshold.
            If ``None`` then will be ignored filtering.
            Default is ``None``.
        scale (int): output scale, resulting coordinates will be multiplied by that constant.
            Default is ``4``.
        input_size (int): model input size. Default is ``512``.

    Returns:
        bounding boxes (np.ndarray with shape [M, 4]) and scores (np.ndarray with shape [M])
    """
    cy, cx = np.where(heatmap > threshold)

    boxes, scores = np.empty((len(cy), 4), dtype=int), np.zeros(len(cx), dtype=float)
    for i, (x, y) in enumerate(zip(cx, cy)):
        scores[i] = heatmap[y, x]

        # x, y in segmentation scales -> upscale outputs
        sx, sy = int(x * scale), int(y * scale)
        w, h = (regr[:, y, x] * input_size).astype(int)
        boxes[i, 0] = sx - w // 2
        boxes[i, 1] = sy - h // 2
        boxes[i, 2] = sx + w // 2
        boxes[i, 3] = sy + h // 2

    return np.array(boxes), np.array(scores)


class COCOFileDataset(Dataset):
    def __init__(self, file, img_dir, output_shape, num_classes, down_ratio, max_objects, transforms=None):
        """
        Args:
            file (str or pathlib.Path): path to a json file with annotations.
            img_dir (str): path to directory with images.
            transforms (albumentations.BasicTransform): transforms apply to images and bounding boxes.
                If `None` then images will be converted torch.Tensor (image will be divided by 255.
                and will be changed order to a [C, W, H]).
                Default is `None`.
        """
        self.file = file
        self.img_dir = img_dir
        self.output_shape = output_shape  # represented as pair: (height, width)
        self.output_height, self.output_width = output_shape
        self.n_classes = num_classes
        self.down_ratio = down_ratio
        self.max_objects = max_objects

        self.transforms = transforms

        self.images, self.categories = load_coco_json(file)
        to_drop = []
        for img_id in self.images.keys():
            cnt_bad = 0
            for bbox in self.images[img_id]["annotations"]:
                if bbox["bbox"][2] == 0 or bbox["bbox"][3] == 0:
                    cnt_bad += 1
            if cnt_bad == len(self.images[img_id]["annotations"]):
                to_drop.append(img_id)
        print(f"Will remove {len(to_drop)} bad images")
        for img_id in to_drop:
            del self.images[img_id]

        self.images_list = sorted(self.images.keys())

        self.class_to_cid = {cls_idx: cat_id for cls_idx, cat_id in enumerate(sorted(self.categories.keys()))}
        self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}

    def __len__(self):
        return len(self.images_list)

    @property
    def num_classes(self):
        return len(self.class_to_cid)

    @property
    def class_labels(self):
        labels = []
        for cls_idx in range(len(self.class_to_cid)):
            labels.append(self.categories[self.class_to_cid[cls_idx]])
        return labels

    def get_class_mapping(self):
        """Information about class mapping.

        Returns:
            List with information about classes (List[Dict[str, Union[int, str]]]).
        """
        # fmt: off
        classes = [
            {
                "class": class_idx,
                "id": category_id,
                "name": self.categories[category_id],
            }
            for class_idx, category_id in self.class_to_cid.items()
        ]
        # fmt: on
        return classes

    def info(self):
        """Information about dataset.

        Returns:
            str with information about dataset
        """
        num_images = len(self.images)
        num_labels = len(self.categories)
        labels_list = sorted(self.categories)
        num_images_with_bboxes = sum(1 for img in self.images.values() if len(img["annotations"]) != 0)
        num_images_without_bboxes = sum(1 for img in self.images.values() if len(img["annotations"]) == 0)
        bbox_cnt = dict(
            Counter(
                self.categories[annot["category_id"]] for img in self.images.values() for annot in img["annotations"]
            ).most_common()
        )
        txt = (
            f"                       Num images: {num_images}\n"
            f"                 Number of labels: {num_labels}\n"
            f"                           Labels: {labels_list}\n"
            f"   Num images with bounding boxes: {num_images_with_bboxes}\n"
            f"Num images without bounding boxes: {num_images_without_bboxes}\n"
            f"          Bounding box statistics: {bbox_cnt}"
        )
        return txt

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.img_dir is not None:
            path = os.path.join(self.img_dir, path)
        image = read_image(path)
        original_size = [image.shape[0], image.shape[1]]  # height, width

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        for annotation in img_record["annotations"]:
            pixel_xywh = annotation["bbox"]
            # skip bounding boxes with 0 height or 0 width
            if pixel_xywh[2] == 0 or pixel_xywh[3] == 0:
                continue
            xyxy = pixels_to_absolute(pixel_xywh, width=img_record["width"], height=img_record["height"])
            assert all(0 <= num <= 1 for num in xyxy), f"All numbers should be in range [0, 1], but got {xyxy}!"
            bbox_class = str(self.cid_to_class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]
        else:
            image = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        labels = np.array([int(items[4]) for items in boxes])
        boxes = np.array([items[:4] for items in boxes], dtype=np.float32)
        # boxes = change_box_order(boxes, "xyxy2xywh")  # (x1, y1, x2, y2) -> (cx, cy, w, h)

        heatmap_height = self.output_height // self.down_ratio
        heatmap_width = self.output_width // self.down_ratio
        # heatmap = np.zeros((self.n_classes, heatmap_height, heatmap_width), dtype=np.float32)
        heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        regr = np.zeros((2, heatmap_height, heatmap_width), dtype=np.float32)
        center = change_box_order(boxes, "xyxy2xywh")

        for x, y, w, h in center:
            a = int(x * heatmap_width) // IN_SCALE
            b = int(y * heatmap_height) // IN_SCALE
            heatmap = draw_msra_gaussian(heatmap, (a, b), sigma=np.clip(w * h, 2, 4))

        regrs = center[:, 2:]

        for r, (x, y, _, _) in zip(regrs, center):
            for i in range(-2, 2 + 1):
                for j in range(-2, 2 + 1):
                    try:
                        # fmt: off
                        a = max(int(x * heatmap_width) // IN_SCALE + i, 0)
                        b = min(int(y * heatmap_height) // IN_SCALE + j, heatmap_height)
                        regr[:, a, b] = r
                        # fmt: on
                    except:  # noqa: E722
                        pass
        regr[0] = regr[0].T
        regr[1] = regr[1].T

        return {
            "path": path,
            "image": image,
            "original_size": original_size,
            "size": [image.size(1), image.size(2)],
            #
            "heatmap": torch.from_numpy(heatmap),
            "regr": torch.from_numpy(regr),
            #
            "boxes": boxes,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch):
        keys = list(batch[0].keys())
        packed_batch = {k: [] for k in keys}
        for element in batch:
            for k in keys:
                packed_batch[k].append(element[k])
        for k in ("image", "heatmap", "regr"):
            packed_batch[k] = torch.stack(packed_batch[k], 0)
        return packed_batch
