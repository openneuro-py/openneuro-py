# Changelog

## 2020.5

- Raise an exception if user `--include`s a path that doesn't exist in the
  dataset. Previously, we would silently ignore this issue.
- Optimze checks whether we need to resume a download or not.
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
