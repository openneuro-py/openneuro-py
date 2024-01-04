from typing import Annotated, Optional

import typer

import openneuro
from openneuro._download import download, login

app = typer.Typer(no_args_is_help=True)


@app.command(name="download")
def download_cli(
    dataset: Annotated[str, typer.Option(help="The OpenNeuro dataset identifier.")],
    tag: Annotated[
        Optional[str], typer.Option(help="The tag (version) of the dataset.")
    ] = None,
    target_dir: Annotated[
        Optional[str], typer.Option(help="The directory to download to.")
    ] = None,
    include: Annotated[
        Optional[str],
        typer.Option(
            help="Only include the specified file or directory. "
            "Can be passed multiple times."
        ),
    ] = None,
    exclude: Annotated[
        Optional[str],
        typer.Option(
            help="Exclude the specified file or directory. "
            "Can be passed multiple times."
        ),
    ] = None,
    verify_hash: Annotated[
        bool,
        typer.Option(help="Whether to check the SHA256 hash of each downloaded file."),
    ] = True,
    verify_size: Annotated[
        bool,
        typer.Option(help="Whether to check the size of each downloaded file."),
    ] = True,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Try the specified number of times to download a file before failing."
        ),
    ] = 5,
    max_concurrent_downloads: Annotated[
        int,
        typer.Option(help="The maximum number of downloads to run in parallel."),
    ] = 5,
) -> None:
    """Download datasets from OpenNeuro."""
    download(
        dataset=dataset,
        tag=tag,
        target_dir=target_dir,
        include=include,
        exclude=exclude,
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
    )


@app.command(name="login")
def login_cli() -> None:
    """Login to OpenNeuro and store an access token."""
    login()


def show_version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"This is openneuro-py {openneuro.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            help="Show the version of openneuro-py.",
            callback=show_version_callback,
            is_eager=True,
        ),
    ] = False,
):
    """Access OpenNeuro datasets."""
    pass
