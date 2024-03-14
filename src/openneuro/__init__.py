"""openneuro-py is a lightweight client for accessing OpenNeuro datasets.

Created and maintained by
Richard Höchenberger <richard.hoechenberger@gmail.com>
"""

from importlib import metadata

try:
    __version__ = metadata.version("openneuro-py")
except metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

from openneuro._download import download as download
from openneuro._download import login as login
