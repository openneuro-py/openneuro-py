"""Glob-style pattern matching for OpenNeuro file paths."""

from collections.abc import Iterable

from wcmatch import glob


def is_dotfile(path: str) -> bool:
    """Check whether any component of *path* starts with a dot."""
    return any(part.startswith(".") for part in path.split("/"))


def glob_filter(
    all_filenames: Iterable[str],
    patterns: Iterable[str],
) -> dict[str, set[str]]:
    """Match filenames against glob patterns, returning per-pattern results.

    Uses ``.gitignore``-style semantics:

    * Bare patterns (no ``/``) match the basename at any depth, so ``*.fif``
      matches ``.fif`` files everywhere, not just at the root.
    * Every pattern is also tried with ``/**`` appended so that directory-like
      patterns (``sub-01``, ``sub-0001/anat``) match all files underneath.
    * A leading ``/`` anchors the pattern to the dataset root and disables
      basename matching.
    * Consistent with ``.gitignore``, ``*`` and ``**`` do **not** match
      dot-prefixed filenames. Use an explicit ``.`` prefix (e.g. ``'**/.*'``)
      to match hidden files.
    """
    filenames = list(all_filenames)
    base_flags = glob.GLOBSTAR
    results: dict[str, set[str]] = {}

    for pattern in patterns:
        original = pattern
        anchored = pattern.startswith("/")
        pattern = pattern.removeprefix("/")
        stripped = pattern.rstrip("/")
        bare = "/" not in stripped

        # Bare, non-anchored patterns match basenames at any depth (MATCHBASE)
        flags = base_flags | glob.MATCHBASE if bare and not anchored else base_flags
        matched: set[str] = {
            str(p) for p in glob.globfilter(filenames, pattern, flags=flags)
        }

        # Always also try as a directory prefix
        matched |= {
            str(p)
            for p in glob.globfilter(filenames, stripped + "/**", flags=base_flags)
        }

        results[original] = matched

    return results
