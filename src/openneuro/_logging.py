import logging
import sys

from tqdm.auto import tqdm

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("openneuro-py")


if hasattr(sys.stdout, "encoding") and sys.stdout.encoding.lower() == "utf-8":
    stdout_unicode = True
elif hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    stdout_unicode = True
else:
    stdout_unicode = False


def log(message: str, cli_only: bool = False) -> None:
    """Emit a log message.

    Parameters
    ----------
    message
        The message to emit.
    cli_only
        Whether to emit the message only when running from the CLI. If `False`, the
        message will shop up when running from the CLI and the Python API.

    """
    from openneuro import _RUNNING_FROM_CLI  # avoid circular import

    if cli_only and not _RUNNING_FROM_CLI:
        return

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
