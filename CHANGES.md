# Changelog

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
