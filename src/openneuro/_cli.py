from typing import Annotated

import typer

import openneuro
from openneuro._config import Source
from openneuro._download import download, login

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(name="download")
def download_cli(
    dataset: Annotated[
        str, typer.Option(help="The OpenNeuro dataset identifier.", show_default=False)
    ],
    tag: Annotated[
        str | None,
        typer.Option(help="The tag (version) of the dataset.", show_default=False),
    ] = None,
    target_dir: Annotated[
        str | None,
        typer.Option(help="The directory to download to.", show_default=False),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            help="Only include the specified file or directory. "
            "Can be passed multiple times.",
            show_default=False,
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            help="Exclude the specified file or directory. "
            "Can be passed multiple times.",
            show_default=False,
        ),
    ] = None,
    verify_hash: Annotated[
        bool,
        typer.Option(
            help="Whether to check the hash of each downloaded file.",
        ),
    ] = True,
    nemar_checksums: Annotated[
        bool,
        typer.Option(
            "--nemar-checksums",
            help="Verify an OpenNeuro download against NEMAR's checksums. "
            "OpenNeuro publishes none of its own, so large (multipart-uploaded) "
            "files are otherwise left unverified. Ignored when --source=nemar, "
            "which always ships checksums.",
        ),
    ] = False,
    verify_size: Annotated[
        bool,
        typer.Option(help="Whether to check the size of each downloaded file."),
    ] = True,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Try the specified number of times to download a file before failing.",
        ),
    ] = 5,
    max_concurrent_downloads: Annotated[
        int,
        typer.Option(
            min=1,
            help="The maximum number of downloads to run in parallel.",
        ),
    ] = 5,
    metadata_timeout: Annotated[
        float,
        typer.Option(
            help="Timeout in seconds for metadata queries.",
        ),
    ] = 15.0,
    source: Annotated[
        Source | None,
        typer.Option(
            help="Where to fetch the data from: OpenNeuro itself, or the NEMAR "
            "mirror (https://nemar.org), which carries the EEG, MEG, and iEEG "
            "datasets published on OpenNeuro. --tag always refers to an "
            "OpenNeuro revision either way. Defaults to the OPENNEURO_SOURCE "
            "environment variable, or 'openneuro'.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Download datasets from OpenNeuro."""
    if nemar_checksums and not verify_hash:
        raise typer.BadParameter(
            "--nemar-checksums cannot be combined with --no-verify-hash: the "
            "first asks for stricter verification, the second for none."
        )
    download(
        dataset=dataset,
        tag=tag,
        target_dir=target_dir,
        include=include,
        exclude=exclude,
        verify_hash="nemar" if nemar_checksums else verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        metadata_timeout=metadata_timeout,
        source=source,
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
        bool | None,
        typer.Option(
            "--version",
            help="Show the version of openneuro-py.",
            callback=show_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Access OpenNeuro datasets."""
    pass
