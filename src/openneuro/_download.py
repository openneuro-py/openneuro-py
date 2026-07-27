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
import contextlib
import dataclasses
import hashlib
import io
import json
import shlex
import string
import sys
import threading
import time
from collections.abc import Coroutine, Iterable
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal, TypeVar

import aiofiles
import niquests
from pydantic import ValidationError
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from openneuro import __version__, _glob
from openneuro._config import get_token, init_config
from openneuro._console import console, cprint
from openneuro._models import DatasetFile, Snapshot

# niquests verifies TLS certificates against the operating system's native
# trust store by default (via wassima), which covers users in enterprise
# environments with custom CAs. No explicit SSL context is required, and the
# same verification applies across the sync and async sessions used below.


def _probe_unicode() -> bool:
    """Whether the stream the console writes to can encode emoji.

    Jupyter takes the `print()` path in `cprint`, but there both streams are
    UTF-8 `OutStream`s, so probing stderr answers for it too. `encoding` is
    typed loosely because it is `None` on a redirected stream such as
    `io.StringIO` (`contextlib.redirect_stderr`), which must not raise here:
    this runs at import time.
    """
    encoding = getattr(sys.stderr, "encoding", None)
    if isinstance(encoding, str) and encoding.lower() == "utf-8":
        return True
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
        return True
    return False


unicode_ok = _probe_unicode()


def login() -> None:
    """Login to OpenNeuro and store an access token."""
    init_config()


_T = TypeVar("_T")

# HTTP server responses that indicate hopefully intermittent errors that
# warrant a retry.
allowed_retry_codes = (408, 500, 502, 503, 504, 522, 524)


class _RetryableError(Exception):
    """Raised inside _attempt_download to signal the caller should retry."""


class _DownloadError(Exception):
    """Terminal per-file download failure.

    Carries a short human-readable reason, the direct download URL (may be
    empty when no URL was available), and a dataset-level debug hint.
    """

    def __init__(self, reason: str, hint: str, url: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.hint = hint
        self.url = url


@dataclasses.dataclass(frozen=True)
class _FileInfo:
    """Per-file metadata collected in `_download_files` before task creation."""

    url: str
    size: int | None
    outfile: Path
    remote_path: str


@dataclasses.dataclass
class _DownloadStats:
    """Running tallies of what was actually fetched over the network.

    Files that already exist locally with a matching size (and hash) are
    skipped, so these counts reflect only the data that was really downloaded.
    """

    n_files: int = 0
    n_bytes: int = 0


allowed_retry_exceptions = (
    # Connection errors (refused/reset/aborted) and DNS failures
    # ("[Errno -3] Temporary failure in name resolution"). Connect timeouts land
    # here too, since ``ConnectTimeout`` is itself a ``ConnectionError``.
    niquests.ConnectionError,
    # Read timeouts are a ``Timeout``, not a ``ConnectionError``, so list them
    # separately.
    niquests.ReadTimeout,
    # Incomplete chunked reads: "peer closed connection without sending
    # complete message body".
    niquests.exceptions.ChunkedEncodingError,
)
user_agent_header: dict[str, str] = {"user-agent": f"openneuro-py/{__version__}"}

_MAX_CONCURRENT_HEAD_REQUESTS = 50

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


_debug_hint_template = string.Template(
    "If this is unexpected:\n\n"
    f"1. Navigate to {gql_url}\n"
    "2. Enter and run the operation: `$query_str`\n"
    '3. In the Response, try to manually download the "urls" for the '
    "failing files listed above.\n\n"
    "If the download fails, open a GitHub issue like "
    "https://github.com/OpenNeuroOrg/openneuro/issues/3145"
)


def _safe_query(
    query: str, *, timeout: float | None = None
) -> tuple[dict[str, Any] | None, bool]:
    cookies: dict[str, str] = {}
    try:
        token = get_token()
        cookies["accessToken"] = token
        cprint("🍪 Using API token to log in")
    except ValueError:
        pass  # No login

    try:
        with niquests.Session(
            # Race IPv6/IPv4 connections (RFC 8305) so a broken stack (e.g.,
            # non-working IPv6) does not stall or fail the request.
            happy_eyeballs=True,
        ) as client:
            # headers/cookies are passed per-request rather than to the Session
            # constructor, which only gained those kwargs in niquests 3.19.
            response = client.post(
                gql_url,
                json={"query": query},
                timeout=timeout,
                headers=user_agent_header,
                cookies=cookies,
            )
    except allowed_retry_exceptions:
        return None, True

    if response.status_code in allowed_retry_codes:
        return None, True

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"GraphQL request failed (HTTP {response.status_code})")

    return response_json, False


def _write_retry(*, what: str, reason: str, retry: int, backoff: float) -> None:
    remaining = "1 retry remains" if retry == 1 else f"{retry} retries remain"
    remaining += f", backing off {backoff:0.1f}s"
    cprint(
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

    raw_snapshots = response_json["data"]["dataset"]["snapshots"]
    tags = [s["id"].replace(f"{dataset_id}:", "") for s in raw_snapshots]

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
) -> Snapshot:
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
        raw = response_json["data"]["dataset"]["latestSnapshot"]
    else:
        raw = response_json["data"]["snapshot"]
    try:
        return Snapshot.model_validate(raw)
    except ValidationError as e:
        sanitized_details = json.dumps(
            e.errors(include_input=False), indent=2, default=str
        )
        raise RuntimeError(
            "The OpenNeuro API returned an unexpected response. "
            "Please open an issue at "
            "https://github.com/openneuro-py/openneuro-py/issues\n\n"
            f"Validation details: {sanitized_details}"
        ) from e


def _retry_request(
    query: str, *, what: str, timeout: float, max_retries: int, retry_backoff: float
) -> dict[str, Any]:
    response_json: dict[str, Any] | None = None
    for retry in reversed(range(max_retries + 1)):
        response_json, request_timed_out = _safe_query(query, timeout=timeout)
        # Sometimes we do get a response, but it contains a gateway timeout error
        # message (504 or 502 status code)
        if response_json is not None and "errors" in response_json:
            error_message = response_json["errors"][0]["message"]
            if (
                error_message.startswith(("504", "502", "connect ECONNREFUSED"))
                or error_message.endswith("due to timeout")
                or error_message == "fetch failed"
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
        error_message = response_json["errors"][0]["message"]
        if error_message == "You do not have access to read this dataset.":
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
            raise RuntimeError(f'Query failed when {what}: "{error_message}"')
    return response_json


async def _download_file(
    *,
    client: niquests.AsyncSession,
    url: str,
    remote_file_size: int | None,
    outfile: Path,
    remote_path: str,
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    retry_backoff: float,
    semaphore: asyncio.Semaphore,
    head_semaphore: asyncio.Semaphore,
    query_str: str,
    progress: Progress,
    overall_task: TaskID,
    stats: _DownloadStats,
) -> None:
    """Download an individual file, retrying on transient errors."""
    try:
        for attempt in range(max_retries + 1):
            try:
                await _attempt_download(
                    client=client,
                    url=url,
                    remote_file_size=remote_file_size,
                    outfile=outfile,
                    remote_path=remote_path,
                    verify_hash=verify_hash,
                    verify_size=verify_size,
                    semaphore=semaphore,
                    head_semaphore=head_semaphore,
                    query_str=query_str,
                    progress=progress,
                    overall_task=overall_task,
                    is_retry=attempt > 0,
                    stats=stats,
                )
                return
            except _RetryableError as err:
                if isinstance(err.__cause__, niquests.Timeout):
                    reason = "Request timed out"
                elif isinstance(err.__cause__, niquests.ConnectionError):
                    reason = "Could not connect (DNS or network error)"
                elif err.__cause__ is not None:
                    reason = str(err.__cause__) or "Error"
                else:
                    reason = str(err) or "Error"
                if attempt < max_retries:
                    _write_retry(
                        what=f"downloading {remote_path}",
                        reason=reason,
                        retry=max_retries - attempt,
                        backoff=retry_backoff,
                    )
                    await asyncio.sleep(retry_backoff)
                    retry_backoff *= 2
                else:
                    attempts = (
                        "1 retry" if max_retries == 1 else f"{max_retries} retries"
                    )
                    raise _DownloadError(
                        reason=f"{reason} (failed after {attempts})",
                        hint=_debug_hint_template.substitute(query_str=query_str),
                        url=url,
                    ) from (err.__cause__ or err)
    except _DownloadError as exc:
        # Report as soon as it is terminal: the end-of-run summary can be
        # hours away on a large dataset.
        cprint(
            _unicode(
                f"Failed to download {remote_path}: {exc.reason}", emoji="❌", end=""
            )
        )
        raise


async def _attempt_download(
    *,
    client: niquests.AsyncSession,
    url: str,
    remote_file_size: int | None,
    outfile: Path,
    remote_path: str,
    verify_hash: bool,
    verify_size: bool,
    semaphore: asyncio.Semaphore,
    head_semaphore: asyncio.Semaphore,
    query_str: str,
    progress: Progress,
    overall_task: TaskID,
    is_retry: bool,
    stats: _DownloadStats,
) -> None:
    """Single download attempt (HEAD → local check → GET)."""
    if outfile.exists():
        local_file_size = outfile.stat().st_size
    else:
        local_file_size = 0
    # For debugging purposes, if there is a problem with a specific file, lines like
    # this can help (used for https://github.com/OpenNeuroOrg/openneuro/issues/3665):
    #
    # cprint(f"Downloading: {outfile.name} from {url}")
    # cprint(f"Query:       {query_str}")
    # if outfile.name == "lh.sphere":
    #     raise RuntimeError(query_str)

    # The OpenNeuro servers are sometimes very slow to respond, so use a
    # gigantic timeout for those. niquests applies the timeout to connecting
    # and reading only; tasks may legitimately wait a long time for a
    # connection from the shared session's pool, and the semaphores are the
    # sole concurrency throttle.
    if url.startswith("https://openneuro.org/crn/"):
        timeout = 60
    else:
        timeout = 5

    # Phase 1: HEAD request to get remote file hash.
    try:
        async with head_semaphore:
            response = await client.head(
                url, headers=user_agent_header, timeout=timeout
            )
            if response.status_code in allowed_retry_codes:
                raise _RetryableError(f"HTTP {response.status_code}")
            if not response.ok:
                raise _DownloadError(
                    reason=f"HEAD request failed with HTTP {response.status_code}",
                    hint=_debug_hint_template.substitute(query_str=query_str),
                    url=url,
                )
            headers = response.headers
    except allowed_retry_exceptions as exc:
        raise _RetryableError from exc

    # Try to get the S3 MD5 hash for the file.
    etag = headers.get("etag")
    etag_hash = etag.strip('"') if etag is not None else None
    remote_file_hash = (
        etag_hash if (etag_hash is not None and len(etag_hash) == 32) else None
    )

    # Phase 2: Local file check (no semaphore held — allows other tasks
    # to use network slots while we do local I/O).
    request_headers: dict[str, str] = user_agent_header.copy()
    request_headers["Accept-Encoding"] = ""  # Disable compression

    mode: Literal["ab", "wb"] = "wb"
    if (
        outfile.exists()
        and remote_file_size is not None
        and local_file_size == remote_file_size
    ):
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
            # On a retry these bytes were streamed (and counted) by an earlier
            # attempt of this run, so discarding them must uncount them too.
            if is_retry:
                progress.update(overall_task, advance=-local_file_size)
            outfile.unlink()
            local_file_size = 0
        else:
            # Download complete, skip.
            if not is_retry:
                progress.update(overall_task, advance=remote_file_size or 0)
            return
    elif (
        outfile.exists()
        and remote_file_size is not None
        and local_file_size < remote_file_size
    ):
        # Download incomplete, resume.
        desc = f"Resuming {outfile.name}"
        request_headers["Range"] = f"bytes={local_file_size}-"
        mode = "ab"
        if not is_retry:
            progress.update(overall_task, advance=local_file_size)
    elif (
        outfile.exists()
        and remote_file_size is not None
        and local_file_size > remote_file_size
    ):
        # Local file is larger than remote – overwrite.
        desc = f"Re-downloading {outfile.name}: file size mismatch."
        if is_retry:
            progress.update(overall_task, advance=-local_file_size)
        outfile.unlink()
        local_file_size = 0
    elif outfile.exists():
        # Remote size unknown – re-download to be safe.
        desc = f"Re-downloading {outfile.name}: remote file size unknown."
        if is_retry:
            progress.update(overall_task, advance=-local_file_size)
        outfile.unlink()
        local_file_size = 0
    else:
        # File doesn't exist locally, download entirely.
        desc = outfile.name

    # Phase 3: GET request to download the file (re-acquires semaphore).
    try:
        async with semaphore:
            response = await client.get(
                url, headers=request_headers, timeout=timeout, stream=True
            )
            # Explicitly close (rather than "async with response") so the
            # connection is released back to the pool on every exit path, while
            # staying compatible with niquests before AsyncResponse became an
            # async context manager (3.19).
            try:
                if response.ok:
                    pass  # All good!
                elif response.status_code in allowed_retry_codes:
                    raise _RetryableError(f"HTTP {response.status_code}")
                else:
                    raise _DownloadError(
                        reason=f"HTTP {response.status_code} when trying to download",
                        hint=_debug_hint_template.substitute(query_str=query_str),
                        url=url,
                    )

                num_bytes = await _retrieve_and_write_to_disk(
                    response=response,
                    outfile=outfile,
                    remote_path=remote_path,
                    mode=mode,
                    desc=desc,
                    local_file_size=local_file_size,
                    remote_file_size=remote_file_size,
                    remote_file_hash=remote_file_hash,
                    verify_hash=verify_hash,
                    verify_size=verify_size,
                    progress=progress,
                    overall_task=overall_task,
                )
            finally:
                await response.close()
    except allowed_retry_exceptions as exc:
        raise _RetryableError from exc

    # The GET completed without raising, so this file was really downloaded.
    stats.n_files += 1
    stats.n_bytes += num_bytes


async def _retrieve_and_write_to_disk(
    *,
    response: niquests.AsyncResponse,
    outfile: Path,
    remote_path: str,
    mode: Literal["ab", "wb"],
    desc: str,
    local_file_size: int,
    remote_file_size: int | None,
    remote_file_hash: str | None,
    verify_hash: bool,
    verify_size: bool,
    progress: Progress,
    overall_task: TaskID,
) -> int:
    """Stream the response to disk, returning the number of bytes downloaded."""
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
        # A transient per-file task that is removed once the file is done, so
        # completed downloads leave no leftover bars behind (gh-323).
        file_task = progress.add_task(
            escape(desc),
            total=remote_file_size,
            completed=local_file_size,
        )
        try:
            downloaded = 0
            # The default chunk_size=-1 yields data as it arrives off the
            # socket (as httpx's aiter_bytes did), which niquests recommends
            # for performance. Compression is disabled (Accept-Encoding: ""),
            # so each chunk's length is the number of raw bytes downloaded.
            async for chunk in await response.iter_content():
                await f.write(chunk)
                progress.update(file_task, advance=len(chunk))
                progress.update(overall_task, advance=len(chunk))
                downloaded += len(chunk)
                if verify_hash:
                    hash.update(chunk)
        finally:
            progress.remove_task(file_task)

        if verify_hash and remote_file_hash is not None:
            got = hash.hexdigest()
            if got != remote_file_hash:
                raise _RetryableError(
                    f"Hash mismatch for {remote_path}: "
                    f"expected {remote_file_hash}, got {got}"
                )

        # Check the file was completely downloaded.
        if verify_size:
            await f.flush()
            local_file_size = outfile.stat().st_size
            if remote_file_size is not None and local_file_size != remote_file_size:
                raise _RetryableError(
                    f"Size mismatch for {remote_path}: expected "
                    f"{remote_file_size} bytes, but downloaded "
                    f"{local_file_size} bytes"
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
                # These bytes are an error blob, never partial data, so drop
                # them: left in place, a retry can mistake the blob for a
                # complete file (its size may match) and skip the re-download.
                progress.update(overall_task, advance=-local_file_size)
                outfile.unlink()
                raise _RetryableError(
                    f"Error downloading {remote_path}: got JSON error response: {data}"
                )

    return downloaded


async def _refresh_progress(progress: Progress, interval: float = 0.1) -> None:
    """Refresh *progress* roughly 10x/second from the event loop.

    rich runs a background thread to auto-refresh progress bars in a terminal,
    but disables it in Jupyter notebooks. Driving `refresh` ourselves on the
    download's own event loop keeps the bars live there. Runs until cancelled.
    """
    while True:
        await asyncio.sleep(interval)
        progress.refresh()


def _make_progress() -> Progress:
    """Create the progress display shown while downloading files.

    A single `rich` progress display drives both the persistent "Overall"
    byte-progress bar and the transient per-file bars, which are removed as
    each file finishes. This avoids the blank lines that per-file `tqdm` bars
    left behind on completion (gh-323). It shares the module-level `console`
    so status messages printed with `cprint` interleave cleanly above it.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


async def _download_files(
    *,
    target_dir: Path,
    files: Iterable[DatasetFile],
    verify_hash: bool,
    verify_size: bool,
    max_retries: int,
    retry_backoff: float,
    max_concurrent_downloads: int,
    query_str: str,
    stats: _DownloadStats,
) -> list[tuple[str, _DownloadError]]:
    """Download files concurrently, returning a list of per-file failures."""
    # Semaphore (counter) to limit maximum number of concurrent download
    # coroutines.
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    # HEAD requests use a separate, higher-limit semaphore so they complete
    # quickly without blocking file downloads.
    head_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_HEAD_REQUESTS)
    normalized_query_str = " ".join(shlex.split(query_str, posix=False))

    # Collect file metadata before creating tasks so the overall progress
    # bar can be created first and passed to each coroutine.
    file_infos: list[_FileInfo] = []
    pre_failures: list[tuple[str, _DownloadError]] = []
    for file in files:
        filename = Path(file.filename)
        if not file.urls:
            pre_failures.append(
                (
                    file.filename,
                    _DownloadError(
                        reason=f"No download URLs for {file.filename}. "
                        "The file may have been removed from the dataset.",
                        hint="",
                    ),
                )
            )
            continue
        url = file.urls[0]

        outfile = target_dir / filename
        outfile.parent.mkdir(parents=True, exist_ok=True)
        file_infos.append(
            _FileInfo(
                url=url,
                size=file.size,
                outfile=outfile,
                remote_path=file.filename,
            )
        )

    total_bytes = sum(fi.size or 0 for fi in file_infos)
    # A single session shared by all download tasks, so open connections are
    # bounded by the pool size instead of the file count. The pool is sized to
    # never throttle below the semaphores: it bounds sockets per host, while
    # the semaphores remain the sole queuing mechanism.
    connection_bound = max_concurrent_downloads + _MAX_CONCURRENT_HEAD_REQUESTS
    async with niquests.AsyncSession(
        pool_connections=connection_bound,
        pool_maxsize=connection_bound,
        # Race IPv6/IPv4 connections (RFC 8305) so a broken stack (e.g.,
        # non-working IPv6) does not stall or fail downloads.
        happy_eyeballs=True,
    ) as client:
        with _make_progress() as progress:
            overall_task = progress.add_task("Overall", total=total_bytes)
            download_tasks = [
                _download_file(
                    client=client,
                    url=fi.url,
                    remote_file_size=fi.size,
                    outfile=fi.outfile,
                    remote_path=fi.remote_path,
                    verify_hash=verify_hash,
                    verify_size=verify_size,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    semaphore=semaphore,
                    head_semaphore=head_semaphore,
                    query_str=normalized_query_str,
                    progress=progress,
                    overall_task=overall_task,
                    stats=stats,
                )
                for fi in file_infos
            ]
            remote_paths = [fi.remote_path for fi in file_infos]
            del file_infos
            # rich disables its background auto-refresh thread in Jupyter, so
            # drive refreshes ourselves on the download's event loop; otherwise
            # the bars would render once and then appear frozen (gh-323).
            refresher = (
                asyncio.ensure_future(_refresh_progress(progress))
                if console.is_jupyter
                else None
            )
            try:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
            finally:
                if refresher is not None:
                    refresher.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await refresher
                    progress.refresh()  # paint the final state

    failures: list[tuple[str, _DownloadError]] = list(pre_failures)
    for remote_path, result in zip(remote_paths, results):
        if isinstance(result, _DownloadError):
            failures.append((remote_path, result))
        elif isinstance(result, asyncio.CancelledError):
            raise result
        elif isinstance(result, BaseException):
            # Unexpected exception — wrap it so it ends up in the summary too.
            # Keep the type: these are bugs rather than download failures, and
            # a bare message like "'urls'" says nothing without its KeyError.
            detail = str(result)
            name = type(result).__name__
            failures.append(
                (
                    remote_path,
                    _DownloadError(
                        reason=f"{name}: {detail}" if detail else name, hint=""
                    ),
                )
            )
    return failures


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
    if unicode_ok:
        msg = f"{emoji} {msg} {end}"
    elif end == "…":
        msg = f"{msg} ..."
    return msg


def _print_download_failures(failures: list[tuple[str, _DownloadError]]) -> None:
    """Print a summary of per-file download failures and raise RuntimeError."""
    n = len(failures)
    noun = "file" if n == 1 else "files"
    msg_fail = (
        f"❌ Failed to download {n} {noun}"
        if unicode_ok
        else f"Failed to download {n} {noun}"
    )
    arrow = "→" if unicode_ok else "->"
    lines = [f"\n{msg_fail}:\n"]
    for remote_path, exc in failures:
        lines.append(f"  {remote_path}")
        lines.append(f"    {arrow} {exc.reason}")
        if exc.url:
            lines.append(f"    {arrow} {exc.url}")
    # The debug hint is dataset-level — print once.
    hint = next((exc.hint for _, exc in failures if exc.hint), None)
    if hint:
        lines.append("")
        lines.append(hint)
    lines.append("")
    lines.append(
        "Re-run this command to retry; already-downloaded files will be skipped."
    )
    cprint("\n".join(lines))
    raise RuntimeError(
        f"Failed to download {n} {noun}. "
        "Re-run this command to retry; already-downloaded files will be skipped."
    )


def _format_size(num_bytes: int) -> str:
    """Return a human-readable size like `0 B`, `12.3 kB`, or `1.2 GB`."""
    size = float(num_bytes)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            break
        size /= 1024
    return f"{num_bytes} {unit}" if unit == "B" else f"{size:.1f} {unit}"


def _run_coroutine_blocking(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run `coroutine` to completion, blocking until it finishes, and return its result.

    When no event loop is running (the usual CLI/script case), we simply own one
    via `asyncio.run`. When a loop is *already* running on this thread — most
    commonly inside a Jupyter notebook — we cannot block it, and `asyncio.run`
    would raise `RuntimeError`. In that case we drive our own event loop in a
    worker thread and join it, so `download` stays synchronous: the files are on
    disk when it returns, the download stats are populated, and any error
    propagates to the caller instead of being swallowed by a fire-and-forget
    task (gh-329).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running, so it is safe to create and drive one here.
        return asyncio.run(coroutine)

    # A loop is already running on this thread; run ours in a separate thread
    # so we can block on it without touching the caller's loop.
    error: dict[str, BaseException] = {}
    result: dict[str, _T] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coroutine)
        except BaseException as exc:  # re-raised on the calling thread below
            error["exc"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "exc" in error:
        # `raise exc` preserves `exc.__traceback__`, so the caller still sees the
        # original worker-thread stack down to the real failure site.
        raise error["exc"]
    return result["value"]


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
        The dataset to retrieve, for example `ds000248`.
    tag
        The tag (revision) of the dataset to retrieve.
    target_dir
        The directory in which to store the downloaded data. If `None`,
        create a subdirectory with the dataset name in the current working
        directory.
    include
        Files and directories to download. **Only** these files and directories
        will be retrieved. Uses glob-style matching: `*` matches any characters
        except `/`, `**` matches across directory boundaries, and `?`
        matches a single non-`/` character. Patterns without a `/` also
        match as directory prefixes (e.g., `'sub-01'` includes all files
        under `sub-01/`, and `'sub-0*'` includes all files under every
        matching directory). Use a leading `/` to restrict to the dataset
        root (e.g., `'/*.json'`). As an example, if you would like to
        download only subject '1' and run '01' files, you can do so via:
        `'sub-1/**/*run-01*'`.

        > **Note:** Consistent with `.gitignore` semantics, `*` and `**` do
        > **not** match dot-prefixed (hidden) filenames. To include such files,
        > use an explicit pattern like `'**/.*'`. The BIDS specification reserves
        > dotfiles for system use, so they are rarely needed.
    exclude
        Files and directories to exclude from downloading.
        Uses the same glob-style matching as `include`.

        > **Note:** Certain essential BIDS metadata files are always downloaded
        > regardless of `exclude` patterns: `dataset_description.json`,
        > `participants.tsv`, `participants.json`, `README`, `CHANGES`, and
        > `.bidsignore`. The dot-prefixed `.bidsignore` is downloaded even
        > though other dotfiles are skipped by default, because BIDS validators
        > rely on it to know which files to ignore.
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

    msg_problems = "problems 🤯" if unicode_ok else "problems"
    msg_bugs = "bugs 🪲" if unicode_ok else "bugs"
    msg_hello = "👋 Hello!" if unicode_ok else "Hello!"
    msg_great_to_see_you = "Great to see you!"
    if unicode_ok:
        msg_great_to_see_you += " 🤗"
    msg_please = "👉 Please" if unicode_ok else "   Please"

    msg = (
        f"\n{msg_hello} This is openneuro-py {__version__}. "
        f"{msg_great_to_see_you}\n\n"
        f"   {msg_please} report {msg_problems} and {msg_bugs} at\n"
        f"      https://github.com/openneuro-py/openneuro-py/issues\n"
    )
    cprint(msg)
    cprint(_unicode(f"Preparing to download {dataset}", emoji="🌍"))

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
    tag = metadata.id.replace(f"{dataset}:", "")
    if target_dir.exists():
        # Once we find the first child, we know the directory is not empty, so we can
        # stop iterating immediately.
        target_dir_empty = next(target_dir.iterdir(), None) is None

        if not target_dir_empty:
            local_tag = _get_local_tag(dataset_id=dataset, dataset_dir=target_dir)

            if local_tag is None:
                cprint(
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
        ".bidsignore",
    }

    all_files = metadata.files
    del metadata
    filenames = [f.filename for f in all_files]

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
    files: list[DatasetFile] = [f for f in all_files if f.filename in keep]

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
        f"Checking {len(files)} files, downloading as needed "
        f"({max_concurrent_downloads} concurrent downloads)."
    )
    cprint(_unicode(msg, emoji="📥", end=""))
    if not console.is_jupyter:
        cprint("")  # Blank line before the progress bars (terminal only)

    query_str = snapshot_query_template.safe_substitute(
        tag=tag or "null",
        dataset_id=dataset,
    )
    stats = _DownloadStats()
    coroutine = _download_files(
        target_dir=target_dir,
        files=files,
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        max_concurrent_downloads=max_concurrent_downloads,
        query_str=query_str,
        stats=stats,
    )

    # Block until the download actually finishes, even when a loop is already
    # running (e.g. in Jupyter), so failures surface and the summary below
    # reports the real stats rather than an empty tally (gh-329).
    failures = _run_coroutine_blocking(coroutine)

    if failures:
        _print_download_failures(failures)

    n_files = stats.n_files
    plural = "file" if n_files == 1 else "files"
    summary = (
        f"Finished downloading {dataset} "
        f"(downloaded {n_files} {plural} and {_format_size(stats.n_bytes)}).\n"
    )
    cprint(_unicode(summary, emoji="✅", end=""))
    cprint(_unicode("Please enjoy your brains.\n", emoji="🧠", end=""))
