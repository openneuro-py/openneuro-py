try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata

try:
    __version__ = metadata.version("openneuro-py")
except metadata.PackageNotFoundError:
    # package is not installed
    pass

from .download import download  # noqa: F401
