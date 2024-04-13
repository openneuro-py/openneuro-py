import logging
import sys

from tqdm.auto import tqdm

# logger = logging.getLogger("openneuro-py")
logger = logging.getLogger()


if hasattr(sys.stdout, "encoding") and sys.stdout.encoding.lower() == "utf-8":
    stdout_unicode = True
elif hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    stdout_unicode = True
else:
    stdout_unicode = False


def log(message: str) -> None:
    from openneuro import _RUNNING_FROM_CLI  # avoid circular import

    if _RUNNING_FROM_CLI:
        logger.log(level=logging.INFO, msg=message)
    else:
        tqdm.write(message)


def _unicode(msg: str, *, emoji: str = " ", end: str = "…") -> str:
    if stdout_unicode:
        msg = f"{emoji} {msg} {end}"
    elif end == "…":
        msg = f"{msg} ..."
    return msg
