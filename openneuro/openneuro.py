"""
openneuro-py is a lightweight client for accessing OpenNeuro datasets.

Created and maintained by
Richard Höchenberger <richard.hoechenberger@gmail.com>
"""

import click

from .download import download_cli
from .config import init_config
from . import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Access OpenNeuro datasets.
    """
    pass


@click.command()
def login_cli(**kwargs):
    """Login to Open Neuro and write the ~/.openneuro config file."""
    init_config()


cli.add_command(download_cli, name='download')
cli.add_command(login_cli, name='login')
