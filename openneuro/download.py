import time
from pathlib import Path
import httpx
import hashlib
from typing import Optional, Tuple

from tqdm import tqdm
import click

from .config import default_base_url


# HTTP server responses that indicate hopefully intermittent errors that
# warrant a retry.
allowed_retry_codes = (408, 500, 502, 503, 504, 522, 524)


def _get_download_metadata(*,
                           base_url: str,
                           dataset_id: str,
                           tag: Optional[str] = None,
                           max_retries: int,
                           retry_backoff: float) -> dict:
    """Retrieve dataset metadata required for the download.
    """
    if tag is None:
        url = f'{base_url}crn/datasets/{dataset_id}/download'
    else:
        url = f'{base_url}crn/datasets/{dataset_id}/snapshots/{tag}/download'

    response = httpx.get(url)
    if 200 <= response.status_code <= 299:
        response_json = response.json()
        return response_json
    elif response.status_code in allowed_retry_codes and max_retries > 0:
        tqdm.write(f'Error {response.status_code}, retrying …')
        time.sleep(retry_backoff)
        max_retries -= 1
        retry_backoff *= 2
        _get_download_metadata(base_url=base_url, dataset_id=dataset_id,
                               tag=tag, max_retries=max_retries,
                               retry_backoff=retry_backoff)
    else:
        raise RuntimeError(f'Error {response.status_code} when trying to '
                           f'fetch metadata.')


def _download_file(*,
                   url: str,
                   api_file_size: int,
                   outfile: Path,
                   verify_hash: bool,
                   verify_size: bool,
                   max_retries: int,
                   retry_backoff: float) -> None:
    """Download an individual file.
    """
    if outfile.exists():
        local_file_size = outfile.stat().st_size
    else:
        local_file_size = 0

    # Check if we need to resume a download
    # The file sizes provided via the API often do not match the sizes reported
    # by the HTTP server. Rely on the sizes reported by the HTTP server.
    with httpx.Client() as client:
        response = client.get(url=url)
    try:
        remote_file_size = int(response.headers['content-length'])
    except KeyError:
        # TSV and JSON files may not have a Content-Length header set.
        remote_file_size = len(response.content)

    headers = {}
    if outfile.exists() and local_file_size == remote_file_size:
        # Download complete, skip.
        tqdm.write(f'Skipping {outfile.name}: already downloaded.')
        return
    elif outfile.exists() and local_file_size < remote_file_size:
        # Download incomplete, resume.
        desc = f'Resuming {outfile.name}'
        headers['Range'] = f'bytes={local_file_size}-'
        mode = 'ab'
    elif outfile.exists():
        # Local file is larger than remote – overwrite.
        desc = f'Re-downloading {outfile.name}: file size mismatch.'
        mode = 'wb'
    else:
        # File doesn't exist locally, download entirely.
        desc = outfile.name
        mode = 'wb'

    with httpx.stream('GET', url=url, headers=headers) as response:
        if 200 <= response.status_code <= 299:
            pass  # All good!
        elif response.status_code in allowed_retry_codes and max_retries > 0:
            tqdm.write(f'Error {response.status_code}, retrying …')
            time.sleep(retry_backoff)
            max_retries -= 1
            retry_backoff *= 2
            _download_file(url=url, remote_file_size=remote_file_size,
                           outfile=outfile, verify_hash=verify_hash,
                           verify_size=verify_size, max_retries=max_retries,
                           retry_backoff=retry_backoff)
        else:
            raise RuntimeError(f'Error {response.status_code} when trying '
                               f'to download {outfile} from {url}')

        hash = hashlib.sha256()
        with tqdm.wrapattr(open(outfile, mode=mode),
                           'write',
                           miniters=1,
                           initial=local_file_size,
                           desc=desc,
                           dynamic_ncols=True,
                           total=remote_file_size) as f:

            for chunk in response.iter_bytes():
                f.write(chunk)
                if verify_hash:
                    hash.update(chunk)

            if verify_hash:
                tqdm.write(f'SHA256 hash: {hash.hexdigest()}')

        # Check the file was completely downloaded.
        if verify_size:
            f.flush()
            local_file_size = outfile.stat().st_size
            if not local_file_size == remote_file_size:
                raise RuntimeError(f'Server claimed file size would be '
                                   f'{remote_file_size} bytes, but downloaded '
                                   f'{local_file_size} byes.')


def _download_files(*,
                    target_dir: Path,
                    files: dict,
                    verify_hash: bool,
                    verify_size: bool,
                    max_retries: int,
                    retry_backoff: float) -> None:
    """Download files, one by one.
    """
    for file in files:
        filename = Path(file['filename'])
        api_file_size = file['size']
        url = file['urls'][0]

        outfile = target_dir / filename
        outfile.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url=url, api_file_size=api_file_size, outfile=outfile,
                       verify_hash=verify_hash, verify_size=verify_size,
                       max_retries=max_retries, retry_backoff=retry_backoff)


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
@click.option('--verify_size', type=bool, default=True, show_default=True,
              help='Whether to check the downloaded file size matches what '
                   'the server announced.')
@click.option('--max_retries', type=int, default=5, show_default=True,
              help='Try the specified number of times to download a file '
                   'before failing.')
def download(*,
             dataset: str,
             tag: Optional[str] = None,
             target_dir: Optional[str] = None,
             include: Optional[Tuple[str]] = None,
             exclude: Optional[Tuple[str]] = None,
             verify_hash: bool = False,
             verify_size: bool = True,
             max_retries: int = 5) -> None:
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
    verify_size
        Whether to check if the downloaded file size matches what the server
        announced.
    max_retries
        Try the specified number of times to download a file before failing.
    """
    if target_dir is None:
        target_dir = Path(dataset)

    include = [] if include is None else include
    exclude = [] if exclude is None else exclude

    retry_backoff = 0.5  # seconds
    metadata = _get_download_metadata(base_url=default_base_url,
                                      dataset_id=dataset,
                                      tag=tag,
                                      max_retries=max_retries,
                                      retry_backoff=retry_backoff)

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

    _download_files(target_dir=target_dir,
                    files=files,
                    verify_hash=verify_hash,
                    verify_size=verify_size,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff)
