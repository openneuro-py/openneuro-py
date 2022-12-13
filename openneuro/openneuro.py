"""
openneuro-py is a lightweight client for accessing OpenNeuro datasets.

Created and maintained by
Richard Höchenberger <richard.hoechenberger@gmail.com>
"""

import click

from ._download import login, download_cli
from . import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Access OpenNeuro datasets.
    """
    pass


@click.command()
def login_cli():
    """Login to OpenNeuro and store an access token."""
    login()


cli.add_command(download_cli, name='download')
cli.add_command(login_cli, name='login')
