from pathlib import Path
import httpx
import hashlib
from typing import Optional, Tuple

from tqdm import tqdm
import click

from .config import default_base_url


def _get_download_metadata(*,
                           base_url: str,
                           dataset_id: str,
                           tag: Optional[str] = None) -> dict:
    """Retrieve dataset metadata required for the download.
    """
    if tag is None:
        url = f'{base_url}crn/datasets/{dataset_id}/download'
    else:
        url = f'{base_url}crn/datasets/{dataset_id}/snapshots/{tag}/download'

    response = httpx.get(url)
    if response.status_code != 200:
        raise RuntimeError(f'Error {response.status_code} when trying to '
                           f'fetch metadata.')

    response_json = response.json()
    return response_json


def _download_files(*,
                    target_dir: Path,
                    files: dict,
                    verify_hash: bool) -> None:
    """Download individual files.
    """
    for file in files:
        filename = Path(file['filename'])
        file_size = file['size']
        url = file['urls'][0]

        outfile = target_dir / filename
        outfile.parent.mkdir(parents=True, exist_ok=True)
        headers = {}

        if outfile.exists():
            local_file_size = outfile.stat().st_size
        else:
            local_file_size = 0

        # Check if we need to resume a download
        if outfile.exists() and local_file_size == file_size:
            # Download complete, skip.
            print(f'Skipping {filename.name}: already downloaded.')
            continue
        elif outfile.exists() and local_file_size < file_size:
            # Download incomplete, resume.
            desc = f'Resuming {filename.name}'
            headers['Range'] = f'bytes={local_file_size}-'
            mode = 'ab'
        elif outfile.exists():
            # Local file is larger than remote – overwrite.
            desc = f'Re-downloading {filename.name}: file size mismatch.'
            mode = 'wb'
        else:
            # File doesn't exist locally, download entirely.
            desc = filename.name
            mode = 'wb'

        with httpx.stream('GET', url=url, headers=headers) as response:
            if response.status_code not in (200, 206):  # OK, Partial Content
                raise RuntimeError(f'Error {response.status_code} when trying '
                                   f'to download {outfile}.')

            hash = hashlib.sha256()
            with tqdm.wrapattr(open(outfile, mode=mode),
                               'write',
                               miniters=1,
                               initial=local_file_size,
                               desc=desc,
                               dynamic_ncols=True,
                               total=file_size) as f:

                for chunk in response.iter_bytes():
                    f.write(chunk)
                    if verify_hash:
                        hash.update(chunk)

                if verify_hash:
                    tqdm.write(f'SHA256 hash: {hash.hexdigest()}')

            # Check the file was completely downloaded.
            f.flush()
            assert outfile.stat().st_size == file_size


@click.command()
@click.option('--dataset', required=True, help='The OpenNeuro dataset name.')
@click.option('--tag', help='The tag (version) of the dataset.')
@click.option('--target_dir', help='The directory to download to.')
@click.option('--include', multiple=True,
              help='Only include the specified file or directory. Can be '
                   'passed multiple times.')
@click.option('--exclude', multiple=True,
              help='Exclude the specified file or directory. Can be passed '
                   'multiple times.')
@click.option('--verify_hash', type=bool, default=False, show_default=True,
              help='Whether to print the SHA256 hash of each downloaded file.')
def download(*,
             dataset: str,
             tag: Optional[str] = None,
             target_dir: Optional[str] = None,
             include: Optional[Tuple[str]] = None,
             exclude: Optional[Tuple[str]] = None,
             verify_hash: bool = False) -> None:
    """Download datasets from OpenNeuro.\f

    Parameters
    ----------
    dataset
        The dataset to retrieve, for example ``ds000248``.
    tag
        The tag (revision) of the dataset to retrieve.
    target_dir
        The directory in which to store the downloaded data. If ``None``,
        create a subdirectory with the dataset name in the current working
        directory.
    include
        Files and directories to download. **Only** these files and directories
        will be retrieved.
    exclude
        Files and directories to exclude from downloading.
    verify_hash
        Whether to calculate and print the SHA256 hash of each downloaded file.
    """
    if target_dir is None:
        target_dir = Path(dataset)

    include = [] if include is None else include
    exclude = [] if exclude is None else exclude

    metadata = _get_download_metadata(base_url=default_base_url,
                                      dataset_id=dataset,
                                      tag=tag)

    files = []
    for file in metadata['files']:
        filename: str = file['filename']  # TODO properly define metadata type

        # Always include essential BIDS files.
        if filename in ('dataset_description.json',
                        'participants.tsv',
                        'README',
                        'CHANGES'):
            files.append(file)
            continue

        if ((not include or any(filename.startswith(i) for i in include)) and
                not any(filename.startswith(e) for e in exclude)):
            files.append(file)

    _download_files(target_dir=target_dir, files=files,
                    verify_hash=verify_hash)
