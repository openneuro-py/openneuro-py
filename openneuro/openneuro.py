"""
openneuro-py is a leightweight client for accessing OpenNeuro datasets.

Created and maintained by
Richard Höchenberger <richard.hoechenberger@gmail.com>
"""

import click

from .download import download


@click.group()
@click.version_option()
def cli():
    """Access OpenNeuro datasets.
    """
    pass


cli.add_command(download)
