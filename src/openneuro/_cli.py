import click

from openneuro import __version__
from openneuro._download import download_cli, login


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Access OpenNeuro datasets."""
    pass


@click.command()
def login_cli() -> None:
    """Login to OpenNeuro and store an access token."""
    login()


cli.add_command(download_cli, name="download")
cli.add_command(login_cli, name="login")
