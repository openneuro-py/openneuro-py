# openneuro-py

A Python client for accessing [OpenNeuro](https://openneuro.org)
datasets.

![openneuro-py in action](https://raw.githubusercontent.com/hoechenberger/openneuro-py/main/openneuro-py.gif)

## Run without installation (uvx)

You can run `openneuro-py` directly without installing it using `uvx`:

```shell
# Download a dataset without installing the package
uvx openneuro-py download --dataset=ds000246

# Get help
uvx openneuro-py --help
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

> **Note:** If you're using `uvx` instead of installing the package, prefix all commands below with `uvx`. For example, `openneuro-py --help` becomes `uvx openneuro-py --help`.

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

## Basic usage – Python interface

```python
import openneuro as on
on.download(dataset='ds000246', target_dir='data/bids')
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and building.

### Setup development environment

```shell
# Clone the repository
git clone https://github.com/hoechenberger/openneuro-py.git
cd openneuro-py

# Install dependencies and create virtual environment
uv sync

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
