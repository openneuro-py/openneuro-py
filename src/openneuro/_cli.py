import click

from openneuro import __version__
from openneuro._download import download, login


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Access OpenNeuro datasets."""
    pass


@click.command()
@click.option("--dataset", required=True, help="The OpenNeuro dataset name.")
@click.option("--tag", help="The tag (version) of the dataset.")
@click.option("--target_dir", help="The directory to download to.")
@click.option(
    "--include",
    multiple=True,
    help="Only include the specified file or directory. Can be "
    "passed multiple times.",
)
@click.option(
    "--exclude",
    multiple=True,
    help="Exclude the specified file or directory. Can be passed " "multiple times.",
)
@click.option(
    "--verify_hash",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to print the SHA256 hash of each downloaded file.",
)
@click.option(
    "--verify_size",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to check the downloaded file size matches what "
    "the server announced.",
)
@click.option(
    "--max_retries",
    type=int,
    default=5,
    show_default=True,
    help="Try the specified number of times to download a file " "before failing.",
)
@click.option(
    "--max_concurrent_downloads",
    type=int,
    default=5,
    show_default=True,
    help="The maximum number of downloads to run in parallel.",
)
def download_cli(**kwargs) -> None:
    """Download datasets from OpenNeuro."""
    download(**kwargs)


@click.command()
def login_cli() -> None:
    """Login to OpenNeuro and store an access token."""
    login()


cli.add_command(download_cli, name="download")
cli.add_command(login_cli, name="login")
