import logging

from tqdm.auto import tqdm

from openneuro import _RUNNING_FROM_CLI

logger = logging.getLogger("openneuro-py")


def log(message: str) -> None:
    if _RUNNING_FROM_CLI:
        logger.log(level=logging.INFO, msg=message)
    else:
        tqdm.write(message)
