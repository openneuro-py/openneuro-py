# Changelog

## 2021.5

- Ramp up timeouts from 5 to 30 seconds for downloads from `openneuro.org`.
- Drop support for Python 3.6. `openneuro-py` now requires Python 3.7 or newer.
  This change makes development easier.

## 2021.4

- Avoid timeouts that would occur in certain situations.
- Don't stall in situations where a download should be retried.

## 2021.3

- Add Unix path expansion for `openneuro.download()`.
- Support OpenNeuro GraphQL API. New dependencies: `sgqlc` and `requests`.
- Fix crash when not passing any `include`s.

## 2021.2

- Fixes Windows.

## 2021.1

- Fixes for Python <3.8.

## 2021.0

- Improved handling of connection timeouts.

## 2020.7

- Performance improvements.
- Verify file hashes by default whenever possible.

## 2020.6

- Ensure we can operate on Python 3.6.
- Ensure non-Unicode terminals (Windows `cmd`, I'm looking at you!) can be
  used too.

## 2020.5

- Raise an exception if user `--include`s a path that doesn't exist in the
  dataset. Previously, we would silently ignore this issue.
- Optimize checks whether we need to resume a download or not.
- Enable simultaneous downloads (defaults to up to 5 concurrent downloads).
- `openneuro.download()` can now be called from Python scripts directly.

## 2020.4

- Don't rely on OpenNeuro API-reported file sizes anymore, but trust the
  HTTP server. This will do away with error messages, and fix bugs when
  downloading TSV and JSON files, which would sometimes end up being
  incorrectly appended.

## 2020.3.1

- Pin `httpx` requirement to `>=0.15`

## 2020.3

- Switch from `requests` to `httpx` library.
- Add `--verify_size` switch to disable file size sanity check.
- Add auto-retry with backoff on intermittent server errors
