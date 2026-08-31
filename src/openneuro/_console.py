"""Shared `rich` console used for all openneuro-py terminal output.

A single, module-level `Console` is shared by the progress bars and the plain
status messages so the two coordinate: messages printed with `cprint` are
rendered *above* any live progress display instead of clobbering it.

The message-formatting helpers live here too (rather than in `_download`) so
that every module which talks to a remote host — `_download` and `_nemar`
alike — can report progress and retries identically without importing one
another.
"""

import io
import sys

from rich.console import Console

# Progress and status are diagnostics, so they belong on stderr: it keeps
# stdout clean for redirection, and matches where `tqdm` drew its bars. Under
# Jupyter, `rich` renders via `display()` and ignores the stream entirely.
console = Console(stderr=True)


def cprint(msg: str = "") -> None:
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
        )
    )
