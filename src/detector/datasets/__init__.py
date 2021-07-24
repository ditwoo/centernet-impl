import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from detector.datasets.coco import COCOFileDataset, pred2box  # noqa: F401
from detector.utils.misc import seed_all


def get_augmentations(img_size, img_mean, img_std):
    """Augmentations for training and validation datasets.

    Args:
        img_size (Tuple[int, int]): image sizes to use, expected pair of values - [width, height].
        img_mean (Tuple[float, float, float]): channelvise mean pixels value, channels order - [R, G, B].
        img_std (Tuple[float, float, float]): channelvise std value, channels order - [R, G, B].

    Returns:
        train and validation augmentations (albu.Compose).
    """
    train = albu.Compose(
        [
            albu.OneOf(
                [
                    albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
                    albu.RandomGamma(),
                    albu.CLAHE(),
                ]
            ),
            albu.RandomBrightnessContrast(brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
            albu.OneOf([albu.Blur(), albu.MotionBlur(), albu.GaussNoise(), albu.ImageCompression(quality_lower=75)]),
            # albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
            albu.RandomSizedBBoxSafeCrop(*img_size, p=0.5),
            albu.Resize(*img_size),
            albu.Normalize(mean=img_mean, std=img_std),
            albu.HorizontalFlip(p=0.1),
            ToTensorV2(),
        ],
        # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
        bbox_params=albu.BboxParams("albumentations"),
    )
    valid = albu.Compose(
        [
            albu.Resize(*img_size),
            albu.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),
        ],
        # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
        bbox_params=albu.BboxParams(format="albumentations"),
    )
    return train, valid


def get_loaders(
    train_annotations,
    train_images_dir,
    train_batch_size,
    train_num_workers,
    #
    valid_annotations,
    valid_images_dir,
    valid_batch_size,
    valid_num_workers,
    #
    image_size,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    """Build training and validation loaders.

    Args:
        train_annotations (str or pathlib.Path): path to training dataset file.
        train_images_dir (str or pathlib.Path): path to directory with training images.
        train_batch_size (int): training batch size.
        train_num_workers (int): number of workers to use for creating training batches.
        valid_annotations (str or pathlib.Path): path to validation dataset file.
        valid_images_dir (str or pathlib.Path): path to directory with validation images.
        valid_batch_size (int): validation batch size.
        valid_num_workers (int): number of workers to use for creating validation batches.
        image_size (Tuple[int, int]): image sizes
        image_mean (Tuple[float, float, float]): channelvise mean pixels value
        image_std (Tuple[float, float, float]): channelvise std value

    Raises:
        ValueError when passed wrong dataset type.

    Returns:
        train dataloader (torch.utils.data.DataLoader) and valid dataloader (torch.utils.data.DataLoader)
    """
    train_augmentations, valid_augmentations = get_augmentations(image_size, image_mean, image_std)

    train_dataset = COCOFileDataset(
        train_annotations, train_images_dir, image_size, 1, 4, 100, transforms=train_augmentations
    )
    # train_dataset = Subset(train_dataset, [i for i in range(train_batch_size * 3)])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=COCOFileDataset.collate_fn,
        worker_init_fn=seed_all,
        prefetch_factor=1,
    )

    valid_dataset = COCOFileDataset(
        valid_annotations, valid_images_dir, image_size, 1, 4, 100, transforms=valid_augmentations
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=valid_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=COCOFileDataset.collate_fn,
        prefetch_factor=1,
    )
    return train_loader, valid_loader
    # return train_loader, train_loader
