from importlib import metadata

try:
    __version__ = metadata.version("openneuro-py")
except metadata.PackageNotFoundError:
    # package is not installed
    pass

from ._download import download, login  # noqa: F401
