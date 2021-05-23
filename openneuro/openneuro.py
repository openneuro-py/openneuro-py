"""
openneuro-py is a lightweight client for accessing OpenNeuro datasets.

Created and maintained by
Richard Höchenberger <richard.hoechenberger@gmail.com>
"""

import click

from .download import download_cli
from . import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Access OpenNeuro datasets.
    """
    pass


cli.add_command(download_cli, name='download')
