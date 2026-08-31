# openneuro-py

A Python client for accessing [OpenNeuro](https://openneuro.org)
datasets.

![openneuro-py in action](https://raw.githubusercontent.com/hoechenberger/openneuro-py/main/openneuro-py.gif)

## Run without installation (uvx)

You can run `openneuro-py` directly without installing it using `uvx`:

```shell
# Download a dataset without installing the package
uvx openneuro-py@latest download --dataset=ds000246

# Get help
uvx openneuro-py@latest --help
```

## Installation into a Python project

Choose one of the following methods:

```shell
# via uv (recommended):
uv add openneuro-py

# via conda:
conda install -c conda-forge openneuro-py

# via pip:
pip install openneuro-py
```

### Optional: Jupyter and IPython support

For enhanced support in Jupyter Lab, Jupyter Notebook, IPython interactive
sessions, and VS Code's interactive Jupyter interface, install `ipywidgets`:

```shell
# via uv:
uv add ipywidgets

# via conda:
conda install -c conda-forge ipywidgets

# via pip:
pip install ipywidgets
```

## Basic usage – command line interface

> **Note:** If you're using `uvx` instead of installing the package, prefix all commands below with `uvx` and
> invoke `openneuro-py@latest` to use the latest released version.
> For example, `openneuro-py --help` becomes `uvx openneuro-py@latest --help`.

### Getting help

```shell
openneuro-py --help
openneuro-py download --help
openneuro-py login --help
```

### Download an entire dataset

```shell
openneuro-py download --dataset=ds000246
```

### Specify a target directory

To store the downloaded files in a specific directory, use the
`--target-dir` switch. The directory will be created if it doesn't exist
already.

```shell
openneuro-py download --dataset=ds000246 \
                      --target-dir=data/bids
```

### Continue an interrupted download

Interrupted downloads will resume where they left off when you run the command
again.

## Advanced usage – command line interface

### Exclude a directory from the download

```shell
openneuro-py download --dataset=ds000246 \
                      --exclude=sub-emptyroom
```

### Download only a single file

```shell
openneuro-py download --dataset=ds000246 \
                      --include=sub-0001/meg/sub-0001_coordsystem.json
```

Note that a few essential BIDS files are **always** downloaded in addition.

### Download or exclude multiple files

`--include` and `--exclude` can be passed multiple times:

```shell
openneuro-py download --dataset=ds000246 \
                      --include=sub-0001/meg/sub-0001_coordsystem.json \
                      --include=sub-0001/meg/sub-0001_acq-LPA_photo.jpg
```

### Use an API token to log in

To download private datasets, you will need an API key that grants you access
permissions. Go to OpenNeuro.org, My Account → Obtain an API Key. Copy the key,
and run:

```shell
openneuro-py login
```

Paste the API key and press return.

### Download from the NEMAR mirror

[NEMAR](https://nemar.org) is a partner archive at UC San Diego that mirrors
the EEG, MEG, and iEEG datasets published on OpenNeuro. The files are
byte-for-byte identical, and NEMAR publishes a checksum for every file, which
`openneuro-py` verifies as it downloads.

```shell
openneuro-py download --dataset=ds004840 --source=nemar
```

To make it the default for every command, set the `OPENNEURO_SOURCE`
environment variable:

```shell
export OPENNEURO_SOURCE=nemar
```

A few things work differently when downloading from NEMAR:

- **Only MEEG datasets are mirrored.** Anything else — and anything NEMAR has
  not mirrored yet — is unavailable, and `openneuro-py` will tell you to use
  `--source=openneuro` instead.
- **`--tag` always means the OpenNeuro revision**, never NEMAR's own version
  number (the two do not correspond). NEMAR keeps only the single snapshot it
  most recently mirrored, so requesting any other revision fails with a message
  naming the one it does have.
- **Restricted datasets are not available**, since NEMAR serves only public
  data and does not use your OpenNeuro API token.
- **A couple of metadata files differ.** NEMAR points `DatasetDOI` at its own
  identifier (recording the OpenNeuro one under `SourceDatasets`), adds a
  `.bidsignore`, and stores the README as `README.md`. The data files
  themselves are unchanged.

## Basic usage – Python interface

```python
import openneuro as on
on.download(dataset='ds000246', target_dir='data/bids')
```

To download from the NEMAR mirror instead, pass `source`:

```python
on.download(dataset='ds004840', target_dir='data/bids', source='nemar')
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and building.

Pre-commit hooks are run through [lefthook](https://lefthook.dev).

### Setup development environment

```shell
# Clone the repository
git clone https://github.com/hoechenberger/openneuro-py.git
cd openneuro-py

# Install dependencies and create virtual environment
uv sync --locked

# Optional: Install pre-commit hooks
uv run lefthook install

# Run tests
uv run pytest

# Run the CLI during development
uv run openneuro-py --help
```

### Building

```shell
# Build the package
uv build
```
