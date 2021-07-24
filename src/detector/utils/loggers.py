import logging

import torch

INITIALIZED_LOGGERS = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Create logger for experiments.

    Args:
        name (str): logger name.
            If function called multiple times same name or
            name which starts with same prefix then will be
            returned initialized logger from the first call.
        log_file (str): file to use for storing logs.
            Default is `None`.
        log_level (int): logging level.
            Default is `logging.INFO`.

    Returns:
        logging.Logger object.
    """
    logger = logging.getLogger(name)
    if name in INITIALIZED_LOGGERS:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in INITIALIZED_LOGGERS:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    INITIALIZED_LOGGERS[name] = True
    return logger
