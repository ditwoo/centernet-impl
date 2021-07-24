import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from detector.utils.box import box_clamp, box_iou, box_nms, box_select, change_box_order, nms_filter  # noqa: F401
from detector.utils.confusion_matrix import ConfusionMatrix  # noqa: F401
from detector.utils.loggers import get_logger  # noqa: F401
from detector.utils.misc import build_args, get_lr, log_metrics, seed_all, t2d, tqdm  # noqa: F401
from detector.utils.registry import Registry  # noqa: F401

OPTIMIZER_REGISTRY = Registry()
OPTIMIZER_REGISTRY.add(optim.Adam)
OPTIMIZER_REGISTRY.add(optim.AdamW)
OPTIMIZER_REGISTRY.add(optim.RMSprop)
OPTIMIZER_REGISTRY.add(optim.SGD)


SCHEDULER_REGISTRY = Registry()
SCHEDULER_REGISTRY.add(lr_scheduler.StepLR)
SCHEDULER_REGISTRY.add(lr_scheduler.CosineAnnealingLR)
SCHEDULER_REGISTRY.add(lr_scheduler.CosineAnnealingWarmRestarts)
SCHEDULER_REGISTRY.add(lr_scheduler.ReduceLROnPlateau)
SCHEDULER_REGISTRY.add(lr_scheduler.CyclicLR)
