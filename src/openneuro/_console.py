"""Shared `rich` console used for all openneuro-py terminal output.

A single, module-level `Console` is shared by the progress bars and the plain
status messages so the two coordinate: messages printed with `cprint` are
rendered *above* any live progress display instead of clobbering it.

The message-formatting helpers live here too (rather than in `_download`) so
that every module which talks to a remote host — `_download` and `_nemar`
alike — can report progress and retries identically without importing one
another.

`cprint` is also where the command line interface and the Python API part ways
(gh-141): the CLI sets `_RUNNING_FROM_CLI` and gets the printed output it
always had, while library callers get `logging` records on the `"openneuro"`
logger, which they can filter, silence, or redirect like any other library's.
That logger ships with a handler of its own and an `INFO` level, because
staying silent by default would hide the very messages the CLI shows. It does
not propagate to the root logger, so a host application's `basicConfig()` does
not print every message a second time; a caller that wants the records routed
elsewhere adds a handler to (or swaps out the handler of) that logger.
"""

import io
import logging
import sys

from rich.console import Console

# Progress and status are diagnostics, so they belong on stderr: it keeps
# stdout clean for redirection, and matches where `tqdm` drew its bars. Under
# Jupyter, `rich` renders via `display()` and ignores the stream entirely.
console = Console(stderr=True)

#: Set to `True` by `openneuro._cli`; unset means we are used as a library.
_RUNNING_FROM_CLI: bool = False

logger = logging.getLogger("openneuro")


def _console_print(msg: str) -> None:
    """Print a message above any active progress display.

    This is the `rich` replacement for `tqdm.write`. Markup and syntax
    highlighting are disabled so that arbitrary text (file paths, URLs, and
    server error bodies) is shown verbatim rather than being reinterpreted as
    `rich` markup.

    In Jupyter, each `console.print` becomes its own output block wrapped in a
    `<pre>` with vertical margins, so consecutive messages render with large
    gaps between them. Plain `print` instead coalesces into a single stdout
    stream (as `tqdm.write` did), keeping the messages tightly spaced. We flush
    explicitly because the download blocks the main thread (see
    `_run_coroutine_blocking`), which otherwise defers the stream flush until
    the cell finishes.
    """
    if console.is_jupyter:
        print(msg, flush=True)
    else:
        console.print(msg, markup=False, highlight=False)


class _ConsoleHandler(logging.Handler):
    """Emit log records through the shared console.

    A plain `StreamHandler` would write straight to `sys.stderr` and tear
    through a live progress display, so library-mode messages take the same
    route as printed ones.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _console_print(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


# A level set before we were imported was a deliberate choice; don't undo it.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
logger.addHandler(_ConsoleHandler())
logger.propagate = False


def cprint(msg: str = "", *, cli_only: bool = False, level: int = logging.INFO) -> None:
    """Report a status message, printing it or logging it as appropriate.

    Parameters
    ----------
    msg
        The message. From the CLI it is printed verbatim above any active
        progress display; otherwise it is logged to the `"openneuro"` logger.
    cli_only
        Whether the message is pure CLI decoration (a greeting, a sign-off)
        that a library caller has no use for, and should be dropped entirely
        when not running from the CLI.
    level
        The level to log at when not running from the CLI. Ignored by the CLI,
        which renders every message identically.

    """
    if _RUNNING_FROM_CLI:
        _console_print(msg)
        return
    if cli_only:
        return
    # A record spanning several lines is unreadable in most logging setups
    # (gh-141), so split it up and drop the blank lines used for spacing.
    for line in msg.splitlines():
        if line.strip():
            logger.log(level, line)


def _probe_unicode() -> bool:
    """Whether the stream the console writes to can encode emoji.

    Jupyter takes the `print()` path in `cprint`, but there both streams are
    UTF-8 `OutStream`s, so probing stderr answers for it too. `encoding` is
    typed loosely because it is `None` on a redirected stream such as
    `io.StringIO` (`contextlib.redirect_stderr`), which must not raise here:
    this runs at import time.
    """
    encoding = getattr(sys.stderr, "encoding", None)
    if isinstance(encoding, str) and encoding.lower() == "utf-8":
        return True
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
        return True
    return False


unicode_ok = _probe_unicode()


def _unicode(msg: str, *, emoji: str = " ", end: str = "…") -> str:
    if unicode_ok:
        msg = f"{emoji} {msg} {end}"
    elif end == "…":
        msg = f"{msg} ..."
    return msg


def _write_retry(*, what: str, reason: str, retry: int, backoff: float) -> None:
    remaining = "1 retry remains" if retry == 1 else f"{retry} retries remain"
    remaining += f", backing off {backoff:0.1f}s"
    cprint(
        _unicode(
            f"{reason} while {what}, retrying ({remaining})",
            emoji="🔄",
        ),
        level=logging.WARNING,
    )
