"""Glob-style pattern matching for OpenNeuro file paths.

Once we require Python 3.13+, ``match()`` can be replaced with
:meth:`pathlib.PurePosixPath.full_match`, which handles ``*`` vs ``**``
correctly. See https://docs.python.org/3.13/library/pathlib.html#pathlib.PurePath.full_match
"""

import functools
import re
from collections.abc import Iterable


@functools.lru_cache
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern into a regex, caching the result."""
    i, n = 0, len(pattern)
    regex = ""
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # **/ matches zero or more directories
                if i + 2 < n and pattern[i + 2] == "/":
                    regex += "(?:.+/)?"
                    i += 3
                else:
                    # ** at end or before non-/ matches everything
                    regex += ".*"
                    i += 2
                continue
            else:
                regex += "[^/]*"
        elif c == "?":
            regex += "[^/]"
        elif c in r".+^${}()|[]\\":
            regex += "\\" + c
        else:
            regex += c
        i += 1
    return re.compile(regex)


def match(filename: str, pattern: str) -> bool:
    """Match a filename against a glob pattern.

    Unlike :func:`fnmatch.fnmatch`, ``*`` does not cross ``/`` boundaries,
    and ``**`` matches zero or more path segments (including the final one).

    A leading ``/`` anchors the pattern to the root of the file tree.  Because
    filenames stored on OpenNeuro never start with ``/``, we strip it here so
    the regex can match against the bare filename.  Combined with ``*`` not
    crossing ``/`` boundaries, this naturally restricts to root-level files.
    """
    if pattern.startswith("/"):
        pattern = pattern[1:]
    return _compile_pattern(pattern).fullmatch(filename) is not None


def expand_patterns(patterns: Iterable[str]) -> list[str]:
    """Auto-expand bare glob patterns to match at any depth.

    Patterns without a ``/`` that contain glob characters (``*``, ``?``, ``[``)
    are prepended with ``**/``, e.g. ``*.tsv`` becomes ``**/*.tsv``. This
    mirrors ``.gitignore`` behavior and preserves backward compatibility with
    older fnmatch-based matching.
    """
    expanded = []
    for p in patterns:
        if "/" not in p and any(c in p for c in "*?["):
            expanded.append(f"**/{p}")
        else:
            expanded.append(p)
    return expanded


def match_include_exclude(
    filename: str,
    *,
    include: Iterable[str],
    exclude: Iterable[str],
) -> tuple[list[bool], list[bool]]:
    """Check if a filename matches an include or exclude pattern."""
    matches_keep = [filename.startswith(i) or match(filename, i) for i in include]
    matches_remove = [filename.startswith(e) or match(filename, e) for e in exclude]
    return matches_keep, matches_remove
