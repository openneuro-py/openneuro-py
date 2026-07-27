"""Shared `rich` console used for all openneuro-py terminal output.

A single, module-level `Console` is shared by the progress bars and the plain
status messages so the two coordinate: messages printed with `cprint` are
rendered *above* any live progress display instead of clobbering it.
"""

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
