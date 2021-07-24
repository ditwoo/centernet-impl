import functools
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

# from coremltools.converters.mil import register_torch_op
# from coremltools.converters.mil.frontend.torch.ops import _get_inputs
# from coremltools.converters.mil.mil import Builder as mb


# # NOTE: required for correct export to CoreML using latest PyTorch version
# @register_torch_op
# def type_as(context, node):
#     inputs = _get_inputs(context, node)
#     context.add(mb.cast(x=inputs[0], dtype="fp32"), node.name)


tqdm = functools.partial(
    tqdm,
    file=sys.stdout,
    bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}",
    leave=True,
)


def log_metrics(logger, metrics, step, title=None, loader=None):
    """Log metrics.

    Args:
        logger (logging.Logger): logger
        metrics (Dict[str, Any]): metrics
        step (int): step/epoch
        title (str): metrics title
        loader (str): loader name
    """
    metric_strings = []

    if title is not None:
        metric_strings.append(title)

    for k, v in metrics.items():
        if k != "confusion_matrix":
            metric_strings.append(f" {k}: {v}")
        else:
            txt_table = "\n".join(" " + " ".join(str(col) for col in row) for row in v["matrix"])
            metric_strings.append(" {}:\n{}".format(k, txt_table))

    if logger is not None:
        logger.info("\n".join(metric_strings))


def build_args(parser):
    """Add required for training arguments.

    Args:
        parser (argparse.ArgumentParser): parser to modify.

    Returns:
        parser (argparse.ArgumentParser) with arguments required for a training.
    """
    # fmt: off
    parser.add_argument("--train", metavar="PATH", type=str, help="path to train dataset.json", required=True)
    parser.add_argument("--train-img-dir", metavar="PATH", type=str, help="path to directory with train images", required=True)  # noqa: E501
    #
    parser.add_argument("--test", metavar="PATH", type=str, help="path to test dataset.json", required=True)
    parser.add_argument("--test-img-dir", metavar="PATH", type=str, help="path to directory with test images", required=True)  # noqa: E501
    #
    parser.add_argument("--batch-size", metavar="NUMBER", type=int, help="batch size to use for training", default=16)
    parser.add_argument("--num-workers", metavar="NUMBER", type=int, help="number of workers to use in dataloader", default=1)  # noqa: E501
    parser.add_argument("--validation-period", metavar="NUMBER", dest="val_period", type=int, help="period (in epochs) for validation", default=1)  # noqa: E501
    parser.add_argument("--num-epochs", metavar="NUMBER", type=int, help="number epochs to train", default=1)
    parser.add_argument("--device", metavar="DEVICE", type=str, help="device to use for training", default="cpu")
    parser.add_argument("--output", metavar="PATH", type=str, help="directory where should be stored .onnx model and info about classes", required=True)  # noqa: E501
    #
    parser.add_argument("--checkpoint", metavar="PATH", type=str, help="path to a trained model", required=False, default="")  # noqa: E501
    parser.add_argument("--lr", metavar="NUMBER", type=float, help="initial learning rate", required=False, default=1e-3)  # noqa: E501
    # fmt: on

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument("--progress", dest="progress", action="store_true")
    feature_parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=False)

    return parser


def get_lr(optimizer):
    """Get learning rate from optimizer.

    Args:
        optimizer (torch.optim.Optimizer): model optimizer

    Returns:
        learning rate (float) if optimizer is instance of torch.optim.Optimizer,
            otherwise `None`
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        return None

    for param_group in optimizer.param_groups:
        return param_group["lr"]


def t2d(tensor, device):
    """Move tensors to a specified device.

    Args:
        tensor (torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor):
            data to move to a device.
        device (str or torch.device): device where should be moved device

    Returns:
        torch.Tensor or Dict[str, torch.Tensor] or List[torch.Tensor] based on `tensor` type.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed=42, deterministic=True, benchmark=True) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed.
            Default is `42`.
        deterministic (bool): flag to use cuda deterministic
            algoritms for computations.
            Default is `True`.
        benchmark (bool): flag to use benchmark option
            to select the best algorithm for computatins.
            Should be used `True` with fixed size
            data (images or similar) for other types of
            data is better to use `False` option.
            Default is `True`.
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = deterministic
    # small speedup
    torch.backends.cudnn.benchmark = benchmark
