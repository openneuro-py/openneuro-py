"""Openneuro download module.

The flow is roughly:

download
  _get_download_metadata
    _check_snapshot_exists
        _safe_query
  _get_local_tag
  _glob.glob_filter
  _download_files
    _download_file
      _attempt_download
        _retrieve_and_write_to_disk
"""

import asyncio
import hashlib
import io
import json
import shlex
import ssl
import string
import sys
import time
import warnings
from collections.abc import Iterable
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal

import aiofiles
import httpx
import requests
from requests.adapters import HTTPAdapter
from sgqlc.endpoint.requests import RequestsEndpoint
from tqdm.auto import tqdm

from openneuro import __version__, _glob
from openneuro._config import get_token, init_config

# Use system trust store for SSL certificates, which is important for users in
# enterprise environments with custom CAs.
# https://truststore.readthedocs.io/en/latest/#using-truststore-with-requests
# https://stackoverflow.com/questions/78219802/use-truststores-sslcontext-with-python-requests-session-object
#
# The SSLContext construction may fail on some platforms (e.g., macOS) even when
# truststore is importable:
# https://github.com/sethmlarson/truststore/issues/167
#
# truststore temporarily sets verify_mode=CERT_NONE on the context during
# wrap_socket(), so a single SSLContext shared across threads triggers spurious
# InsecureRequestWarnings from urllib3.  Each TruststoreAdapter therefore gets
# its own context; only the httpx path (async, single-threaded) keeps a
# module-level instance.
_use_truststore = True
try:
    import truststore

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except (ImportError, OSError, ssl.SSLError) as exc:
    _use_truststore = False
    ssl_context = ssl.create_default_context()
    warnings.warn(
        f"Could not use truststore for SSL verification ({exc!r}); "
        "falling back to Python/OpenSSL default certificate verification.",
        stacklevel=1,
    )


class TruststoreAdapter(HTTPAdapter):
    def __init__(self, **kwargs: Any) -> None:
        if _use_truststore:
            self._ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        else:
            self._ssl_context = ssl.create_default_context()
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs.setdefault("ssl_context", self._ssl_context)
        return super().init_poolmanager(
            connections,
            maxsize,
            block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs.setdefault("ssl_context", self._ssl_context)
        return super().proxy_manager_for(proxy, **proxy_kwargs)


if hasattr(sys.stdout, "encoding") and sys.stdout.encoding.lower() == "utf-8":
    stdout_unicode = True
elif isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
    stdout_unicode = True
else:
    stdout_unicode = False


def login() -> None:
    """Login to OpenNeuro and store an access token."""
    init_config()


# HTTP server responses that indicate hopefully intermittent errors that
# warrant a retry.
allowed_retry_codes = (408, 500, 502, 503, 504, 522, 524)


class _RetryableError(Exception):
    """Raised inside _attempt_download to signal the caller should retry."""


allowed_retry_exceptions = (
    # For file downloads
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.ConnectError,  # [Errno -3] Temporary failure in name resolution
    # For GraphQL requests via sgqlc (doesn't support httpx)
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
    # "peer closed connection without sending complete message body
    #  (incomplete chunked read)"
    httpx.RemoteProtocolError,
)
user_agent_header: dict[str, str] = {"user-agent": f"openneuro-py/{__version__}"}

# GraphQL endpoint and queries.

gql_url = "https://openneuro.org/crn/graphql"

dataset_query_template = string.Template(
    """
    query {
        dataset(id: "$dataset_id") {
            latestSnapshot {
                id
                files(recursive: true) {
                    filename
                    urls
                    size
                    id
                }
            }
        }
    }
"""
)

all_snapshots_query_template = string.Template(
    """
    query {
        dataset(id: "$dataset_id") {
            snapshots {
                id
            }
        }
    }
"""
)

snapshot_query_template = string.Template(
    """
    query {
        snapshot(datasetId: "$dataset_id", tag: "$tag") {
            id
            files(recursive: true) {
                filename
                urls
                size
                id
            }
        }
    }
"""
)


def _safe_query(
    query: str, *, timeout: float | None = None
) -> tuple[dict[str, Any] | None, bool]:
    with requests.Session() as session:
        session.mount("https://", TruststoreAdapter())
        session.headers.update(user_agent_header)
        try:
            token = get_token()
            session.cookies.set_cookie(
                requests.cookies.create_cookie("accessToken", token)  # type: ignore[no-untyped-call]
            )
            tqdm.write("🍪 Using API token to log in")
        except ValueError:
            pass  # No login
        gql_endpoint = RequestsEndpoint(url=gql_url, session=session, timeout=timeout)
        try:
            response_json = gql_endpoint(query=query)
            request_timed_out = False
        except allowed_retry_exceptions:
            response_json = None
            request_timed_out = True
    return response_json, request_timed_out


def _write_retry(*, what: str, reason: str, retry: int, backoff: float) -> None:
    remaining = "1 retry remains" if retry == 1 else f"{retry} retries remain"
    remaining += f", backing off {backoff:0.1f}s"
    tqdm.write(
        _unicode(
            f"{reason} while {what}, retrying ({remaining})",
            emoji="🔄",
        )
    )


def _check_snapshot_exists(
    *, dataset_id: str, tag: str, max_retries: int, retry_backoff: float
) -> None:
    query = all_snapshots_query_template.substitute(dataset_id=dataset_id)
    response_json = _retry_request(
        query,
        what="fetching list of snapshots",
        timeout=60.0,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )

    snapshots = response_json["data"]["dataset"]["snapshots"]
    tags = [s["id"].replace(f"{dataset_id}:", "") for s in snapshots]

    if tag not in tags:
        raise RuntimeError(
            f'The requested snapshot with the tag "{tag}" '
            f"does not exist for dataset {dataset_id}. "
            f"Existing tags: {', '.join(tags)}"
        )


def _get_download_metadata(
    *,
    dataset_id: str,
    tag: str | None = None,
    max_retries: int,
    retry_backoff: float = 0.5,
    metadata_timeout: float = 15.0,
) -> dict[str, Any]:
    """Retrieve dataset metadata required for the download."""
    if tag is None:
        query = dataset_query_template.substitute(dataset_id=dataset_id)
    else:
        _check_snapshot_exists(
            dataset_id=dataset_id,
            tag=tag,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        query = snapshot_query_template.substitute(dataset_id=dataset_id, tag=tag)

    response_json = _retry_request(
        query,
        what=f"retrieving metadata for {dataset_id}",
        timeout=metadata_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    if tag is None:
        out = response_json["data"]["dataset"]["latestSnapshot"]
    else:
        out = response_json["data"]["snapshot"]
    assert isinstance(out, dict)
    return out


def _retry_request(
    query: str, *, what: str, timeout: float, max_retries: int, retry_backoff: float
) -> dict[str, Any]:
    response_json: dict[str, Any] | None = None
    for retry in reversed(range(max_retries + 1)):
        response_json, request_timed_out = _safe_query(query, timeout=timeout)
        # Sometimes we do get a response, but it contains a gateway timeout error
        # message (504 or 502 status code)
        if response_json is not None and "errors" in response_json:
            message = response_json["errors"][0]["message"]
            if (
                message.startswith(("504", "502", "connect ECONNREFUSED"))
                or message.endswith("due to timeout")
                or message == "fetch failed"
            ):
                request_timed_out = True
        if not request_timed_out:
            break
        if retry > 0:
            _write_retry(
                what=what,
                reason="Request timed out",
                retry=retry,
                backoff=retry_backoff,
            )
            time.sleep(retry_backoff)
            retry_backoff *= 2
    else:
        raise RuntimeError(f"Timeout when {what}.")
    if response_json is None:
        raise RuntimeError(f"Error when {what}.")
    assert isinstance(response_json, dict)
    if "errors" in response_json:
        msg = response_json["errors"][0]["message"]
        if msg == "You do not have access to read this dataset.":
            try:
                # Do we have an API token?
                get_token()
                raise RuntimeError(
                    "We were not permitted to download "
                    f"this dataset ({what}). Perhaps your user "
                    "does not have access to it, or "
                    "your API token is wrong."
                )
            except ValueError as e:
                # We don't have an API token.
                raise RuntimeError(
                    "It seems that this is a restricted "
                    f"dataset ({what}). However, your API token is "
                    "not configured properly, so we could "
                    f"not log you in. {e}"
                )
        else:
            raise RuntimeError(f'Query failed when {what}: "{msg}"')
    return response_json


async def _download_file(
    *,
    url: str,
    api_file_size: int,
    outfile: Path,
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    retry_backoff: float,
    semaphore: asyncio.Semaphore,
    query_str: str,
) -> None:
    """Download an individual file, retrying on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            await _attempt_download(
                url=url,
                api_file_size=api_file_size,
                outfile=outfile,
                verify_hash=verify_hash,
                verify_size=verify_size,
                semaphore=semaphore,
                query_str=query_str,
            )
            return
        except _RetryableError as err:
            if attempt < max_retries:
                if isinstance(err.__cause__, httpx.TimeoutException):
                    reason = "Request timed out"
                elif err.__cause__ is not None:
                    reason = str(err.__cause__) or "Error"
                else:
                    reason = str(err) or "Error"
                _write_retry(
                    what=f"downloading {outfile}",
                    reason=reason,
                    retry=max_retries - attempt,
                    backoff=retry_backoff,
                )
                await asyncio.sleep(retry_backoff)
                retry_backoff *= 2
            else:
                raise RuntimeError(
                    f"Failed to download {outfile} from {url} "
                    f"after {max_retries} retries."
                ) from (err.__cause__ or err)


async def _attempt_download(
    *,
    url: str,
    api_file_size: int,
    outfile: Path,
    verify_hash: bool,
    verify_size: bool,
    semaphore: asyncio.Semaphore,
    query_str: str,
) -> None:
    """Single download attempt (HEAD → local check → GET)."""
    if outfile.exists():
        local_file_size = outfile.stat().st_size
    else:
        local_file_size = 0
    # For debugging purposes, if there is a problem with a specific file, lines like
    # this can help (used for https://github.com/OpenNeuroOrg/openneuro/issues/3665):
    #
    # tqdm.write(f"Downloading: {outfile.name} from {url}")
    # tqdm.write(f"Query:       {query_str}")
    # if outfile.name == "lh.sphere":
    #     raise RuntimeError(query_str)

    # The OpenNeuro servers are sometimes very slow to respond, so use a
    # gigantic timeout for those.
    if url.startswith("https://openneuro.org/crn/"):
        timeout = 60
    else:
        timeout = 5

    async with httpx.AsyncClient(timeout=timeout, verify=ssl_context) as client:
        # Phase 1: HEAD request to get remote file hash and size.
        # The file sizes provided via the API often do not match the sizes
        # reported by the HTTP server. Rely on the HTTP server sizes.
        try:
            async with semaphore:
                response = await client.head(url, headers=user_agent_header)
                headers = response.headers
        except allowed_retry_exceptions as exc:
            raise _RetryableError from exc

        # Try to get the S3 MD5 hash for the file.
        try:
            etag_hash = headers["etag"].strip('"')
            if len(etag_hash) == 32:
                remote_file_hash = etag_hash
            else:  # It's not an MD5 hash.
                remote_file_hash = None
        except KeyError:
            remote_file_hash = None

        # Get the Content-Length.
        try:
            remote_file_size = int(response.headers["content-length"])
        except KeyError:
            # The server doesn't always set a Content-Length header.
            remote_file_size = api_file_size
        if remote_file_size != api_file_size:
            tqdm.write(
                _unicode(
                    f"Warning: size mismatch for {outfile.name}: "
                    f"API size {api_file_size} bytes, "
                    f"server size {remote_file_size} bytes.",
                    emoji="⚠️",
                )
            )

        # Phase 2: Local file check (no semaphore held — allows other tasks
        # to use network slots while we do local I/O).
        request_headers: dict[str, str] = user_agent_header.copy()
        request_headers["Accept-Encoding"] = ""  # Disable compression

        mode: Literal["ab", "wb"] = "wb"
        if outfile.exists() and local_file_size == remote_file_size:
            hash_ = hashlib.md5()

            if verify_hash and remote_file_hash is not None:
                async with aiofiles.open(outfile, "rb") as f:
                    while True:
                        data = await f.read(65536)
                        if not data:
                            break
                        hash_.update(data)

            if (
                verify_hash
                and remote_file_hash is not None
                and hash_.hexdigest() != remote_file_hash
            ):
                desc = f"Re-downloading {outfile.name}: file hash mismatch."
                outfile.unlink()
                local_file_size = 0
            else:
                # Download complete, skip.
                tqdm.write(f"Skipping {outfile.name}: already downloaded.")
                return
        elif outfile.exists() and local_file_size < remote_file_size:
            # Download incomplete, resume.
            desc = f"Resuming {outfile.name}"
            request_headers["Range"] = f"bytes={local_file_size}-"
            mode = "ab"
        elif outfile.exists():
            # Local file is larger than remote – overwrite.
            desc = f"Re-downloading {outfile.name}: file size mismatch."
            outfile.unlink()
            local_file_size = 0
        else:
            # File doesn't exist locally, download entirely.
            desc = outfile.name

        # Phase 3: GET request to download the file (re-acquires semaphore).
        try:
            async with semaphore:
                async with client.stream(
                    "GET", url=url, headers=request_headers
                ) as response:
                    if not response.is_error:
                        pass  # All good!
                    elif response.status_code in allowed_retry_codes:
                        raise _RetryableError(f"HTTP {response.status_code}")
                    else:
                        raise RuntimeError(
                            f"Error {response.status_code} when trying to "
                            f"download {outfile}. If this is unexpected:\n\n"
                            "1. Navigate to "
                            "https://openneuro.org/crn/graphql\n"
                            "2. Enter and run the operation: "
                            f"`{query_str}`\n"
                            "3. In the Response, try to manually download "
                            f'the "urls" for "{outfile.name}", which should '
                            f"contain {url}\n\n"
                            "If the download fails, open a GitHub issue like "
                            "https://github.com/OpenNeuroOrg/openneuro/"
                            "issues/3145"
                        )

                    await _retrieve_and_write_to_disk(
                        response=response,
                        outfile=outfile,
                        mode=mode,
                        desc=desc,
                        local_file_size=local_file_size,
                        remote_file_size=remote_file_size,
                        remote_file_hash=remote_file_hash,
                        verify_hash=verify_hash,
                        verify_size=verify_size,
                    )
        except allowed_retry_exceptions as exc:
            raise _RetryableError from exc


async def _retrieve_and_write_to_disk(
    *,
    response: httpx.Response,
    outfile: Path,
    mode: Literal["ab", "wb"],
    desc: str,
    local_file_size: int,
    remote_file_size: int,
    remote_file_hash: str | None,
    verify_hash: bool,
    verify_size: bool,
) -> None:
    hash = hashlib.md5()

    # If we're resuming a download, ensure the already-downloaded
    # parts of the file are fed into the hash function before
    # we continue.
    if verify_hash and local_file_size > 0:
        async with aiofiles.open(outfile, "rb") as f:
            while True:
                data = await f.read(65536)
                if not data:
                    break
                hash.update(data)

    async with aiofiles.open(outfile, mode=mode) as f:
        with tqdm(
            desc=desc,
            initial=local_file_size,
            total=remote_file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress:
            num_bytes_downloaded = response.num_bytes_downloaded
            async for chunk in response.aiter_bytes():
                await f.write(chunk)
                progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                num_bytes_downloaded = response.num_bytes_downloaded
                if verify_hash:
                    hash.update(chunk)

        if verify_hash and remote_file_hash is not None:
            got = hash.hexdigest()
            if got != remote_file_hash:
                raise RuntimeError(
                    f"Hash mismatch for:\n{outfile}\n"
                    f"Expected:\n{remote_file_hash}\nGot:\n{got}"
                )

        # Check the file was completely downloaded.
        if verify_size:
            await f.flush()
            local_file_size = outfile.stat().st_size
            if local_file_size != remote_file_size:
                raise RuntimeError(
                    f"Server claimed size of {outfile} would be "
                    f"{remote_file_size} bytes, but downloaded "
                    f"{local_file_size} bytes."
                )
    # Secondary check: try loading as JSON for "error" entry
    # We can get for invalid files sometimes the contents:
    # {"error": "an unknown error occurred accessing this file"}
    # This is a 58-byte file, but let's be tolerant and try loading
    # anything less than 200 as JSON and detect a dict with a single
    # "error" entry.
    if verify_size and local_file_size < 200:
        try:
            data = json.loads(outfile.read_text("utf-8"))
        except Exception:
            pass
        else:
            if isinstance(data, dict) and list(data) == ["error"]:
                raise RuntimeError(
                    f"Error downloading:\n{outfile}:\n"
                    f"Got JSON error response contents:\n{data}"
                )


async def _download_files(
    *,
    target_dir: Path,
    files: Iterable[dict[str, Any]],
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    retry_backoff: float,
    max_concurrent_downloads: int,
    query_str: str,
) -> None:
    """Download files, one by one."""
    # Semaphore (counter) to limit maximum number of concurrent download
    # coroutines.
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    download_tasks = []
    normalized_query_str = " ".join(shlex.split(query_str, posix=False))

    for file in files:
        filename = Path(file["filename"])
        api_file_size = file["size"]
        if not file.get("urls"):
            raise RuntimeError(
                f"No download URLs for {filename}. The file may have been "
                "removed from the dataset."
            )
        url = file["urls"][0]

        outfile = target_dir / filename
        outfile.parent.mkdir(parents=True, exist_ok=True)
        download_task = _download_file(
            url=url,
            api_file_size=api_file_size,
            outfile=outfile,
            verify_hash=verify_hash,
            verify_size=verify_size,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            semaphore=semaphore,
            query_str=normalized_query_str,
        )
        download_tasks.append(download_task)

    await asyncio.gather(*download_tasks)


def _get_local_tag(*, dataset_id: str, dataset_dir: Path) -> str | None:
    """Get the local dataset revision."""
    local_json_path = dataset_dir / "dataset_description.json"
    if not local_json_path.exists():
        return None

    local_json_file_content = local_json_path.read_text(encoding="utf-8")
    if not local_json_file_content:
        return None

    local_json = json.loads(local_json_file_content)

    if "DatasetDOI" not in local_json:
        raise RuntimeError(
            'Local "dataset_description.json" does not contain '
            '"DatasetDOI" field. Are you sure this is the '
            "correct directory?"
        )

    local_doi = local_json["DatasetDOI"]
    assert isinstance(local_doi, str)
    if local_doi.startswith("doi:"):
        # Remove the "protocol" prefix
        local_doi = local_doi[4:]

    expected_doi_start = f"10.18112/openneuro.{dataset_id}.v"

    if not local_doi.startswith(expected_doi_start):
        raise RuntimeError(
            f"The existing dataset in the target directory "
            f"appears to be different from the one you "
            f'requested to download. "DatasetDOI" field in '
            f'local "dataset_description.json": '
            f"{local_json['DatasetDOI']}. "
            f"Requested dataset: {dataset_id}"
        )

    local_version = local_doi.replace(f"10.18112/openneuro.{dataset_id}.v", "")
    return local_version


def _unicode(msg: str, *, emoji: str = " ", end: str = "…") -> str:
    if stdout_unicode:
        msg = f"{emoji} {msg} {end}"
    elif end == "…":
        msg = f"{msg} ..."
    return msg


def download(
    *,
    dataset: str,
    tag: str | None = None,
    target_dir: Path | str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    verify_hash: bool = True,
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 5,
    metadata_timeout: float = 15.0,
) -> None:
    """Download datasets from OpenNeuro.

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
        will be retrieved. Uses glob-style matching: ``*`` matches any characters
        except ``/``, ``**`` matches across directory boundaries, and ``?``
        matches a single non-``/`` character. Patterns without a ``/`` also
        match as directory prefixes (e.g., ``'sub-01'`` includes all files
        under ``sub-01/``, and ``'sub-0*'`` includes all files under every
        matching directory). Use a leading ``/`` to restrict to the dataset
        root (e.g., ``'/*.json'``). As an example, if you would like to
        download only subject '1' and run '01' files, you can do so via:
        ``'sub-1/**/*run-01*'``.

        .. note::
            Consistent with ``.gitignore`` semantics, ``*`` and ``**`` do **not**
            match dot-prefixed (hidden) filenames. To include such files, use an
            explicit pattern like ``'**/.*'``. The BIDS specification reserves
            dotfiles for system use, so they are rarely needed.
    exclude
        Files and directories to exclude from downloading.
        Uses the same glob-style matching as ``include``.

        .. note::
            Certain essential BIDS metadata files are always downloaded
            regardless of ``exclude`` patterns: ``dataset_description.json``,
            ``participants.tsv``, ``participants.json``, ``README``, and
            ``CHANGES``.
    verify_hash
        Whether to calculate and verify the MD5 hash of each downloaded file.
    verify_size
        Whether to check if the downloaded file size matches what the server
        announced.
    max_retries
        Try the specified number of times to download a file before failing.
    max_concurrent_downloads
        The maximum number of downloads to run in parallel.
    metadata_timeout
        Timeout in seconds for metadata queries.

    """
    if max_concurrent_downloads < 1:
        raise ValueError("max_concurrent_downloads must be at least 1.")

    msg_problems = "problems 🤯" if stdout_unicode else "problems"
    msg_bugs = "bugs 🪲" if stdout_unicode else "bugs"
    msg_hello = "👋 Hello!" if stdout_unicode else "Hello!"
    msg_great_to_see_you = "Great to see you!"
    if stdout_unicode:
        msg_great_to_see_you += " 🤗"
    msg_please = "👉 Please" if stdout_unicode else "   Please"

    msg = (
        f"\n{msg_hello} This is openneuro-py {__version__}. "
        f"{msg_great_to_see_you}\n\n"
        f"   {msg_please} report {msg_problems} and {msg_bugs} at\n"
        f"      https://github.com/openneuro-py/openneuro-py/issues\n"
    )
    tqdm.write(msg)
    tqdm.write(_unicode(f"Preparing to download {dataset}", emoji="🌍"))

    if target_dir is None:
        target_dir = Path(dataset)
    else:
        target_dir = Path(target_dir)
    target_dir = target_dir.expanduser().resolve()

    include = [include] if isinstance(include, str) else include
    include = [] if include is None else list(include)

    exclude = [exclude] if isinstance(exclude, str) else exclude
    exclude = [] if exclude is None else list(exclude)

    retry_backoff = 0.5  # seconds
    metadata = _get_download_metadata(
        dataset_id=dataset,
        tag=tag,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        metadata_timeout=metadata_timeout,
    )
    del tag
    tag = metadata["id"].replace(f"{dataset}:", "")
    if target_dir.exists():
        # Once we find the first child, we know the directory is not empty, so we can
        # stop iterating immediately.
        target_dir_empty = next(target_dir.iterdir(), None) is None

        if not target_dir_empty:
            local_tag = _get_local_tag(dataset_id=dataset, dataset_dir=target_dir)

            if local_tag is None:
                tqdm.write(
                    "Cannot determine local revision of the dataset, "
                    "and the target directory is not empty. If the "
                    "download fails, you may want to try again with a "
                    "fresh (empty) target directory."
                )
            elif local_tag != tag:
                raise FileExistsError(
                    f"You requested to download revision {tag}, but "
                    f"revision {local_tag} exists locally in the designated "
                    f"target directory. Please either remove this dataset or "
                    f"specify a different target directory, and try again."
                )

    essential_files = {
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "README",
        "CHANGES",
    }

    all_files = metadata["files"]
    del metadata
    filenames = [f["filename"] for f in all_files]

    if include:
        included = _glob.glob_filter(filenames, include)
        included_set = {f for matches in included.values() for f in matches}
    else:
        included_set = {f for f in filenames if not _glob.is_dotfile(f)}

    if exclude:
        excluded = _glob.glob_filter(filenames, exclude)
        excluded_set = {f for matches in excluded.values() for f in matches}
    else:
        excluded_set = set()

    keep = (included_set - excluded_set) | (essential_files & set(filenames))
    files: list[dict[str, Any]] = [f for f in all_files if f["filename"] in keep]

    if include:
        for pattern, matches in included.items():
            if not matches:
                has_glob = any(c in pattern for c in "*?[")
                maybe = [] if has_glob else get_close_matches(pattern, filenames)
                if maybe:
                    extra = (
                        "Perhaps you mean one of these paths:\n- "
                        + "\n- ".join(maybe)
                        + "\n"
                    )
                else:
                    extra = "There were no similar filenames found in the metadata. "
                raise RuntimeError(
                    f"Could not find path in the dataset:\n- {pattern}\n{extra}"
                    "Please check your includes."
                )

    msg = (
        f"Retrieving up to {len(files)} files "
        f"({max_concurrent_downloads} concurrent downloads)."
    )
    tqdm.write(_unicode(msg, emoji="📥", end=""))

    query_str = snapshot_query_template.safe_substitute(
        tag=tag or "null",
        dataset_id=dataset,
    )
    coroutine = _download_files(
        target_dir=target_dir,
        files=files,
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        max_concurrent_downloads=max_concurrent_downloads,
        query_str=query_str,
    )

    # Try to re-use event loop if it already exists. This is required e.g.
    # for use in Jupyter notebooks.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coroutine)
    except RuntimeError:
        asyncio.run(coroutine)

    tqdm.write(_unicode(f"Finished downloading {dataset}.\n", emoji="✅", end=""))
    tqdm.write(_unicode("Please enjoy your brains.\n", emoji="🧠", end=""))
