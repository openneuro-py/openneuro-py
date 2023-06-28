# Changelog

## 2023.1.0 (unreleased)

- Better handling of server response errors.
- We switched from using the unmaintained `appdirs` to `platformdirs`. If you're using private OpenNeuro repositories on macOS, you may have to enter your API tokens again.

## 2022.2.0

- Support latest OpenNeuro API.
- Display suggestions in the exception when `include` contains invalid
  entries.
- Drop list of default excludes. OpenNeuro has fixed server response for the
  respective datasets, so excluding files by default is not necessary anymore.
- Add ability to use an API token to access restricted datasets.

## 2022.1.0

- Fix handling of DOIs that start with `doi:`, as found e.g. in `ds002778`.

## 2021.10.1

- New release for PyPI.

## 2021.10

- Fix unicode terminal detection for Sphinx documentation builds.

## 2021.9

- Add basic support for running Jupyter Notebooks / Jupyter Lab, interactive
  IPython sessions, and in the VS Code interactive Jupyter window.

- Don't crash if the local `dataset_description.json` file is empty when trying
  to resume an aborted download.

- We now by default exclude certain files from the download that are known to
  be invalid for specific datasets. Once the datasets have been fixed on
  OpenNeuro, we will revert these exclusions.

## 2021.8

- Retry downloads if a `ReadError` has occurred.

## 2021.7

- Before resuming a download, check if local and remote datasets actually
  match.

- If neither the API nor the HTTP server provide the file size, don't download
  the file just for size determination. This won't allow us to display progress
  bars anymore in this case, but it will avoid downloading the same file twice.
  Unfortunately, resuming such downloads will be impossible, too: we'll simply
  re-download the files. Note that this only applies to files hosted on
  OpenNeuro.org for which we cannot determine the file size.

- Massively reduce memory footprint (by 2 orders of magnitude).

## 2021.6

- Disable transfer encoding (i.e., compression). This allows for an easier
  check whether a file has been completely downloaded.

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
