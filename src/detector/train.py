import argparse
import json
import os

import numpy as np
import torch
from mean_average_precision import MetricBuilder

from detector.criterions import CenterLoss
from detector.datasets import get_loaders, pred2box
from detector.models import ResNetCenterNet
from detector.utils import (
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    ConfusionMatrix,
    build_args,
    get_logger,
    get_lr,
    log_metrics,
    nms_filter,
    seed_all,
    t2d,
    tqdm,
)

LOGGER = None
IMG_SIZE = None
IMG_MEAN = None
IMG_STD = None


def train_fn(loader, model, device, criterion, optimizer, scheduler=None, verbose=True):
    """Train model on a specified loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader.
        model (torch.nn.Module): model to use for training.
        device (str/int/torch.device): device to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): batch scheduler
            (will be triggered after each batch).
            Default is `None`.
        verbose (bool): option to print information about training progress.
            Default is `True`.

    Returns:
        Metrics (dict where key (str) is a metric name and value (float))
        collected during the training on specified loader
    """
    model.train()
    num_batches = len(loader)
    metrics = {"mask": 0.0, "regr": 0.0, "loss": 0.0}
    progress_str = "mask - {:.4f}, regr - {:.4f}, loss - {:.5f}"
    with tqdm(desc="training", total=num_batches, disable=not verbose) as progress:
        for batch in loader:
            imgs, heatmaps, regrs = t2d((batch["image"], batch["heatmap"], batch["regr"]), device)

            hm, reg = model(imgs)
            # preds = torch.cat((hm, reg), 1)

            loss, mask_loss, regr_loss = criterion(hm, reg, heatmaps, regrs)

            optimizer.zero_grad()
            loss.backward()

            mask_loss_value = mask_loss.item()
            regr_loss_value = regr_loss.item()
            loss_value = loss.item()

            metrics["mask"] += mask_loss_value
            metrics["regr"] += regr_loss_value
            metrics["loss"] += loss_value

            progress.set_postfix_str(progress_str.format(mask_loss_value, regr_loss_value, loss_value))

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            progress.update(1)
    metrics = {k: (v / num_batches) for k, v in metrics.items()}
    metrics["lr"] = get_lr(optimizer)
    return metrics


@torch.no_grad()
def valid_fn(loader, model, device, criterion, num_classes, class_labels, verbose=True, threshold=0.5):
    """Validate model on a specified loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader.
        model (torch.nn.Module): model to use for training.
        device (str/int/torch.device): device to use for training.
        num_classes (int): number of classes in dataset.
        class_labels (List[str]): class labels.
        verbose (bool): option to print information about training progress.
            Default is `True`.

    Returns:
        Metrics (dict where key (str) is a metric name and value (float))
        collected during the validation on specified loader.
    """
    model.eval()
    num_batches = len(loader)
    mean_ap = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    conf_matrix = ConfusionMatrix(num_classes)
    metrics = {"mask": 0.0, "regr": 0.0, "loss": 0.0}
    progress_str = "mask - {:.4f}, regr - {:.4f}, loss - {:.5f}"
    with tqdm(desc="validation", total=num_batches, disable=not verbose) as progress:
        for batch in loader:
            imgs, heatmaps, regrs = t2d((batch["image"], batch["heatmap"], batch["regr"]), device)
            batch_size = imgs.size(0)

            hm, reg = model(imgs)
            # preds = torch.cat((hm, reg), 1)

            loss, mask_loss, regr_loss = criterion(hm, reg, heatmaps, regrs)

            mask_loss_value = mask_loss.item()
            regr_loss_value = regr_loss.item()
            loss_value = loss.item()

            metrics["mask"] += mask_loss_value
            metrics["regr"] += regr_loss_value
            metrics["loss"] += loss_value

            hm = hm.sigmoid()
            pooled = torch.nn.functional.max_pool2d(hm, kernel_size=(3, 3), stride=1, padding=1)
            hm = hm * torch.logical_and(hm >= threshold, pooled >= threshold).float()

            hm_numpy = hm.detach().cpu().numpy()
            reg_numpy = reg.detach().cpu().numpy()

            for i in range(batch_size):
                sample_boxes = []
                sample_classes = []
                sample_scores = []
                for cls_idx in range(hm_numpy.shape[1]):
                    # build predictions
                    cls_boxes, cls_scores = pred2box(
                        hm_numpy[i, cls_idx], reg_numpy[i], threshold=0, scale=4, input_size=512
                    )

                    # skip empty label predictions
                    if cls_scores.shape[0] == 0:
                        continue

                    cls_boxes = cls_boxes / 512.0

                    cls_boxes, cls_classes, cls_scores = nms_filter(
                        cls_boxes, np.full(len(cls_scores), cls_idx), cls_scores, iou_threshold=0.25
                    )
                    sample_boxes.append(cls_boxes)
                    sample_classes.append(cls_classes)
                    sample_scores.append(cls_scores)

                # skip empty predictions
                if len(sample_boxes) == 0:
                    continue

                sample_boxes = np.concatenate(sample_boxes, 0)
                sample_classes = np.concatenate(sample_classes, 0)
                sample_scores = np.concatenate(sample_scores, 0)

                pred_sample = np.concatenate([sample_boxes, sample_classes[:, None], sample_scores[:, None]], -1)
                pred_sample = pred_sample.astype(np.float32)

                # build ground truth
                sample_gt_bboxes = batch["boxes"][i]
                sample_gt_classes = batch["labels"][i]
                gt_sample = np.zeros((sample_gt_classes.shape[0], 7), dtype=np.float32)
                gt_sample[:, :4] = sample_gt_bboxes
                gt_sample[:, 4] = sample_gt_classes

                # update metrics statistics
                mean_ap.add(pred_sample, gt_sample)
                conf_matrix.add(pred_sample, gt_sample[:, :5])

            progress.set_postfix_str(progress_str.format(mask_loss_value, regr_loss_value, loss_value))
            progress.update(1)

    metrics = {k: (v / num_batches) for k, v in metrics.items()}
    metrics["mAP"] = mean_ap.value()["mAP"]
    metrics["confusion_matrix"] = {
        "matrix": conf_matrix.value().tolist(),
        "labels": class_labels,
    }
    return metrics


def experiment(device, args=None):
    """Train model.

    Args:
        device (str): device to use for training.
        args (dict): experiment arguments.
    """
    global LOGGER, IMG_SIZE, IMG_MEAN, IMG_STD

    args = dict() if args is None else args
    #
    verbose = args["progress"]
    #
    architecture = "centernet-resnet18"
    IMG_SIZE = (512, 512)
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)
    #
    logdir = args["output"]
    num_epochs = args["num_epochs"]
    validation_period = args["val_period"]
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    #
    LOGGER = get_logger(architecture, os.path.join(logdir, "experiment.log"))
    LOGGER.info(f"Experiment arguments:\n{json.dumps(args, indent=4)}")

    #######################################################################
    # datasets
    #######################################################################
    train_loader, valid_loader = get_loaders(
        train_annotations=args["train"],
        train_images_dir=args["train_img_dir"],
        train_batch_size=args["batch_size"],
        train_num_workers=args["num_workers"],
        #
        valid_annotations=args["test"],
        valid_images_dir=args["test_img_dir"],
        valid_batch_size=args["batch_size"],
        valid_num_workers=args["num_workers"],
        #
        image_size=IMG_SIZE,
        image_mean=IMG_MEAN,
        image_std=IMG_STD,
    )

    LOGGER.info("Train dataset information:\n" + train_loader.dataset.info())
    LOGGER.info("Validation dataset information:\n" + valid_loader.dataset.info())
    class_labels = valid_loader.dataset.class_labels
    num_classes = valid_loader.dataset.num_classes

    #######################################################################
    # experiment parts
    #######################################################################
    seed_all(42)
    model = ResNetCenterNet(num_classes=num_classes, model_name="resnet18", bilinear=True)
    if args["checkpoint"]:
        state = torch.load(args["checkpoint"], map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        LOGGER.info("Loaded model state from '{}'".format(args["checkpoint"]))
    model = model.to(device)
    criterion = CenterLoss(num_classes, regr_loss_weight=10)
    optimizer = OPTIMIZER_REGISTRY["AdamW"](model.parameters(), **{"lr": args["lr"]})
    epoch_scheduler = SCHEDULER_REGISTRY["CosineAnnealingWarmRestarts"](optimizer, **{"T_0": args["num_epochs"]})
    batch_scheduler = None

    #######################################################################
    # train loop
    #######################################################################
    for epoch_idx in range(1, num_epochs + 1):
        LOGGER.info(f"Epoch: {epoch_idx}/{num_epochs}")
        # casual training loop
        train_metrics = train_fn(train_loader, model, device, criterion, optimizer, batch_scheduler, verbose=verbose)
        log_metrics(LOGGER, train_metrics, epoch_idx, "\nTrain:", loader="train")
        # do validation, if required
        if epoch_idx % validation_period == 0:
            valid_metrics = valid_fn(
                valid_loader, model, device, criterion, num_classes, class_labels, verbose=verbose, threshold=0.25
            )
            log_metrics(LOGGER, valid_metrics, epoch_idx, "\nValidation:", loader="validation")

        # change lr after epoch
        epoch_scheduler.step()

    #######################################################################
    # experiment artifacts
    #######################################################################
    path = os.path.join(logdir, "latest.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
        path,
    )
    LOGGER.info(f"Saved model last state to '{path}'")


def main(args):
    """Training entrypoint.

    Args:
        args (dict): training arguments, expected dict with structure:
            {
                "progress": <bool>,      # option to show progress bars during the training or validation
                "train": <str>,          # path to .json with train data
                "train_img_dir": <str>,  # path to directory with train images
                "test": <str>,           # path to .json with test data
                "test_img_dir": <str>,   # path to directory with test images
                "batch_size": <int>,
                "num_workers": <int>,
                "num_classes": <int>,
                "val_period": <int>,     # validation period (in epochs)
                "num_epochs": <int>,
                "output": <str>,         # directory where will be stored outputs
                "device": <str>,         # "cpu"/"cuda"/"cuda:N" device to use
            }
    """
    device = args.pop("device")
    experiment(device, args)


if __name__ == "__main__":
    parser = build_args(argparse.ArgumentParser())
    args = vars(parser.parse_args())
    main(args)
