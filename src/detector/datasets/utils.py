import json

import cv2
import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]  # noqa: E203
    y, x = np.ogrid[-m : m + 1, -n : n + 1]  # noqa: E203

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]  # noqa: E203
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]  # noqa: E203
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def load_coco_json(path):
    """Read json with annotations.

    Args:
        path (str): path to .json file

    Raises:
        RuntimeError if .json file has no images
        RuntimeError if .json file has no categories

    Returns:
        images mapping and categories mapping
    """

    with open(path, "r") as in_file:
        content = json.load(in_file)

    if not len(content["images"]):
        raise RuntimeError(f"There is no image records in '{path}' file!")

    if not len(content["categories"]):
        raise RuntimeError(f"There is no categories in '{path}' file!")

    images = {}  # image_id -> {file_name, height, width, annotations([{id, iscrowd, category_id, bbox}, ...])}
    for record in content["images"]:
        images[record["id"]] = {
            "file_name": record["file_name"],
            "height": record["height"],
            "width": record["width"],
            "annotations": [],
        }

    categories = {}  # category_id -> name
    for record in content["categories"]:
        categories[record["id"]] = record["name"]

    for record in content["annotations"]:
        images[record["image_id"]]["annotations"].append(
            {
                "id": record["id"],
                "iscrowd": record["iscrowd"],
                "category_id": record["category_id"],
                "bbox": record["bbox"],
            }
        )

    return images, categories


def read_image(path):
    """Read image from given path.

    Args:
        path (str or Path): path to an image.

    Raises:
        FileNotFoundError when missing image file

    Returns:
        np.ndarray with image.
    """
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"There is no '{path}'!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pixels_to_absolute(box, width, height):
    """Convert pixel coordinates to absolute scales ([0,1]).

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x, y, w, h).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x1, y1, x2, y2).
    """
    x, y, w, h = box
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def absolute_to_pixels(box, width, height):
    """Convert absolute coordinates to pixel values.

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x1, y1, x2, y2).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x, y, w, h).
    """
    x1, y1, x2, y2 = box
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]
