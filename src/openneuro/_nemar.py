"""Support for downloading OpenNeuro datasets from the NEMAR mirror.

NEMAR (<https://nemar.org>) is a partner archive at UC San Diego that mirrors
every publicly released EEG, MEG, and iEEG dataset from OpenNeuro. The bytes it
serves are the same bytes OpenNeuro serves; only the way they are addressed
differs, so this module's job is to translate NEMAR's data API into the same
`Snapshot` that `_download._get_download_metadata` builds from OpenNeuro's
GraphQL API. Everything downstream — globbing, resuming, hashing, retrying —
is shared.

Three differences matter enough to be visible to callers:

1. **Coverage.** Only MEEG datasets are mirrored, so a dataset that exists on
   OpenNeuro may legitimately be absent here (`_NotFound` → a `RuntimeError`
   naming `source="openneuro"` as the way out).
2. **Versioning.** NEMAR assigns its *own* version numbers, which do not
   correspond to OpenNeuro's tags: NEMAR `v1.0.2` of `on002578` mirrors
   OpenNeuro's `1.1.0`, and a dataset can be re-released on NEMAR without any
   upstream change. `tag` therefore always means the *OpenNeuro* tag, and it is
   resolved through the `IsDerivedFrom` DOI that NEMAR records in its
   dataset-level metadata. NEMAR keeps only the snapshot it last mirrored, so
   older OpenNeuro revisions are simply unavailable here.
3. **Checksums.** Unlike OpenNeuro, NEMAR publishes a per-file checksum in its
   manifest, which `_download` verifies directly instead of falling back to
   guessing an MD5 out of the S3 `ETag`.
"""

import json
import re
import time
from typing import Any

import niquests

from openneuro._console import _write_retry, cprint
from openneuro._http import allowed_retry_codes, allowed_retry_exceptions
from openneuro._http import user_agent_header as user_agent_header
from openneuro._models import ChecksumAlgorithm, DatasetFile, Snapshot

#: Canonical host for NEMAR's data API. `api.nemar.org/data/` is an alias.
NEMAR_DATA_URL = "https://data.nemar.org"

#: OpenNeuro dataset IDs (``ds<6 digits>``) map onto NEMAR's mirrored IDs
#: (``on<the same 6 digits>``). NEMAR-native datasets use an ``nm`` prefix and
#: have no OpenNeuro counterpart, so they are not reachable through this
#: package.
_DATASET_ID_RE = re.compile(r"^ds(?P<number>\d+)$")

#: The DOI NEMAR records to state which OpenNeuro revision a mirror came from,
#: e.g. ``10.18112/openneuro.ds004840.v1.0.1``.
_OPENNEURO_DOI_RE = re.compile(
    r"^10\.18112/openneuro\.(?P<dataset_id>ds\d+)\.v(?P<tag>.+)$"
)


class _NotFound(Exception):
    """Raised when NEMAR answers a metadata request with HTTP 404."""


def to_nemar_id(dataset_id: str) -> str:
    """Translate an OpenNeuro dataset ID into its NEMAR counterpart.

    Parameters
    ----------
    dataset_id
        An OpenNeuro dataset ID, for example `ds004840`.

    Returns
    -------
    The corresponding NEMAR dataset ID, for example `on004840`.

    Raises
    ------
    ValueError
        When `dataset_id` is not a well-formed OpenNeuro dataset ID.

    """
    match = _DATASET_ID_RE.match(dataset_id)
    if match is None:
        raise ValueError(
            f"Cannot download {dataset_id!r} from NEMAR: only OpenNeuro dataset "
            'IDs of the form "ds<number>" (e.g. ds004840) are mirrored there.'
        )
    return f"on{match['number']}"


def _get_json(
    url: str,
    *,
    what: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> Any:
    """GET a JSON document from NEMAR, retrying transient failures.

    Mirrors the retry behaviour of `_download._retry_request` so that a flaky
    connection behaves the same whichever source is in use.
    """
    response: niquests.Response | None = None
    for retry in reversed(range(max_retries + 1)):
        request_timed_out = False
        try:
            with niquests.Session(
                # Race IPv6/IPv4 connections (RFC 8305) so a broken stack (e.g.,
                # non-working IPv6) does not stall or fail the request.
                happy_eyeballs=True,
            ) as client:
                response = client.get(url, timeout=timeout, headers=user_agent_header)
        except allowed_retry_exceptions:
            request_timed_out = True
        else:
            if response.status_code in allowed_retry_codes:
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

    assert response is not None  # a non-timed-out iteration always sets it
    if response.status_code == 404:
        raise _NotFound(url)
    if not response.ok:
        raise RuntimeError(f"Error when {what}: HTTP {response.status_code}.")
    try:
        return response.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            f"The NEMAR API returned a non-JSON response when {what}. "
            "Please open an issue at "
            "https://github.com/openneuro-py/openneuro-py/issues"
        )


def _openneuro_tag_from_metadata(metadata: Any) -> str | None:
    """Extract the OpenNeuro tag a NEMAR mirror was derived from.

    NEMAR records this as an `IsDerivedFrom` related identifier pointing at the
    versioned OpenNeuro DOI. Returns `None` when the record is missing or does
    not look like an OpenNeuro DOI, which callers treat as "provenance unknown"
    rather than as an error.
    """
    if not isinstance(metadata, dict):
        return None
    for identifier in metadata.get("related_identifiers") or []:
        if not isinstance(identifier, dict):
            continue
        if identifier.get("relation_type") != "IsDerivedFrom":
            continue
        match = _OPENNEURO_DOI_RE.match(str(identifier.get("identifier", "")))
        if match is not None:
            return match["tag"]
    return None


def _files_from_manifest(
    manifest: Any, *, nemar_id: str, version: str
) -> list[DatasetFile]:
    """Convert a NEMAR manifest into `DatasetFile`s."""
    if not isinstance(manifest, list):
        raise RuntimeError(
            f"The NEMAR manifest for {nemar_id} {version} was not a list. "
            "Please open an issue at "
            "https://github.com/openneuro-py/openneuro-py/issues"
        )

    files: list[DatasetFile] = []
    for entry in manifest:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = str(entry["path"])
        # `url` is a pre-signed S3 link that expires after an hour, which is
        # not long enough for a large dataset. `bytes_url` is the stable
        # address that mints a fresh pre-signed link on every request (or, for
        # small git-stored files, points straight at raw.githubusercontent.com),
        # so it is the one that survives a long or resumed download.
        url = entry.get("bytes_url") or entry.get("url")
        files.append(
            DatasetFile(
                filename=path,
                urls=[str(url)] if url else None,
                size=entry.get("size"),
                # NEMAR has no per-file opaque ID; the path is unique already.
                id=path,
                checksum=entry.get("checksum"),
                checksum_algorithm=entry.get("checksum_algorithm"),
            )
        )
    return files


def get_metadata(
    *,
    dataset_id: str,
    tag: str | None,
    max_retries: int,
    retry_backoff: float,
    metadata_timeout: float,
) -> Snapshot:
    """Retrieve the metadata needed to download `dataset_id` from NEMAR.

    Parameters
    ----------
    dataset_id
        The OpenNeuro dataset ID, for example `ds004840`.
    tag
        The requested **OpenNeuro** tag (revision), or `None` for whichever
        revision NEMAR currently mirrors.
    max_retries
        How many times to retry a timed-out metadata request.
    retry_backoff
        Seconds to wait before the first retry; doubled on each subsequent one.
    metadata_timeout
        Per-request timeout in seconds.

    Returns
    -------
    A `Snapshot` whose `id` is `"<dataset_id>:<openneuro_tag>"`, matching what
    the OpenNeuro path returns so that the rest of the download is identical.

    """
    nemar_id = to_nemar_id(dataset_id)

    try:
        versions_doc = _get_json(
            f"{NEMAR_DATA_URL}/{nemar_id}/",
            what=f"retrieving NEMAR versions for {dataset_id}",
            timeout=metadata_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
    except _NotFound:
        raise RuntimeError(_not_mirrored_message(dataset_id))

    version = (
        (versions_doc or {}).get("latest") if isinstance(versions_doc, dict) else None
    )
    if not version:
        raise RuntimeError(
            f"NEMAR lists {dataset_id} (as {nemar_id}) but has not published "
            "any version of it yet, so there is nothing to download. Pass "
            'source="openneuro" to download it from OpenNeuro instead.'
        )

    # NEMAR's dataset-level metadata describes its latest version, which is the
    # one we are about to download.
    try:
        metadata_doc = _get_json(
            f"{NEMAR_DATA_URL}/{nemar_id}/metadata.json",
            what=f"retrieving NEMAR metadata for {dataset_id}",
            timeout=metadata_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
    except _NotFound:
        metadata_doc = None
    openneuro_tag = _openneuro_tag_from_metadata(metadata_doc)

    if openneuro_tag is None:
        # Without the provenance record we cannot honour, or even report, an
        # OpenNeuro revision. That is fatal only if one was actually requested.
        if tag is not None:
            raise RuntimeError(
                f"NEMAR does not record which OpenNeuro revision of "
                f"{dataset_id} it mirrors, so revision {tag} cannot be "
                'requested from it. Pass source="openneuro" to download that '
                "revision from OpenNeuro instead."
            )
        cprint(
            f"NEMAR does not record which OpenNeuro revision of {dataset_id} "
            f"it mirrors; reporting its own version ({version}) instead."
        )
        reported_tag = version
    elif tag is not None and tag != openneuro_tag:
        raise RuntimeError(
            f"NEMAR does not provide revision {tag} of {dataset_id}.\n\n"
            f"NEMAR mirrors a single snapshot per dataset, and currently holds "
            f"OpenNeuro revision {openneuro_tag} (as NEMAR version {version}). "
            f'Either request tag="{openneuro_tag}", or pass '
            'source="openneuro" to download revision '
            f"{tag} from OpenNeuro instead."
        )
    else:
        reported_tag = openneuro_tag

    try:
        manifest = _get_json(
            f"{NEMAR_DATA_URL}/{nemar_id}/{version}/manifest.json",
            what=f"retrieving NEMAR file manifest for {dataset_id}",
            timeout=metadata_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
    except _NotFound:
        raise RuntimeError(
            f"NEMAR has no file manifest for {dataset_id} version {version}. "
            'Pass source="openneuro" to download it from OpenNeuro instead.'
        )

    files = _files_from_manifest(manifest, nemar_id=nemar_id, version=version)
    if not files:
        raise RuntimeError(
            f"The NEMAR manifest for {dataset_id} version {version} lists no "
            'files. Pass source="openneuro" to download it from OpenNeuro '
            "instead."
        )

    return Snapshot(id=f"{dataset_id}:{reported_tag}", files=files)


def _not_mirrored_message(dataset_id: str) -> str:
    """Explain that NEMAR does not carry `dataset_id`."""
    return (
        f"Dataset {dataset_id} is not available from NEMAR.\n\n"
        "NEMAR mirrors the EEG, MEG, and iEEG datasets published on OpenNeuro, "
        "so datasets of other modalities — and datasets that have not been "
        'mirrored yet — are not available there. Pass source="openneuro" to '
        "download it from OpenNeuro instead."
    )


#: Files NEMAR rewrites while mirroring, whose checksums therefore describe
#: NEMAR's copy rather than OpenNeuro's and must never be used to verify an
#: OpenNeuro download.
#:
#: NEMAR repoints ``DatasetDOI`` at its own identifier and records the OpenNeuro
#: one under ``SourceDatasets``, which changes the file. Measured across
#: ds000117, ds000246, and ds004840, this is the *only* difference: of 709
#: git-hashed and 87 md5-hashed files compared against OpenNeuro, every other
#: one matched byte-for-byte.
_REWRITTEN_BY_NEMAR = frozenset({"dataset_description.json"})


def get_checksums(
    *,
    dataset_id: str,
    tag: str,
    max_retries: int,
    retry_backoff: float,
    metadata_timeout: float,
) -> dict[str, tuple[str, ChecksumAlgorithm]]:
    """Fetch NEMAR's checksums for verifying an *OpenNeuro* download.

    OpenNeuro publishes no checksums of its own; `_download` falls back to the
    S3 `ETag`, which is an MD5 only for single-part uploads. Multipart uploads
    return a `"<hash>-<parts>"` digest instead, so the very largest files —
    the ones most likely to arrive truncated — currently go unverified. Because
    NEMAR mirrors OpenNeuro byte-for-byte, its manifest can supply the missing
    checksums.

    This is a best-effort enhancement layered on top of an OpenNeuro download,
    so every failure path degrades to the normal `ETag` behaviour rather than
    raising: NEMAR being unreachable must not break a download that OpenNeuro
    is happily serving.

    Parameters
    ----------
    dataset_id
        The OpenNeuro dataset ID, for example `ds000246`.
    tag
        The OpenNeuro revision being downloaded. Checksums are returned only if
        NEMAR mirrors exactly this revision; a mirror that has fallen behind
        describes different bytes and is refused.
    max_retries
        How many times to retry a timed-out metadata request.
    retry_backoff
        Seconds to wait before the first retry; doubled on each subsequent one.
    metadata_timeout
        Per-request timeout in seconds.

    Returns
    -------
    A mapping of dataset-relative path to `(checksum, algorithm)`, empty when
    NEMAR cannot vouch for this revision.

    """
    try:
        snapshot = get_metadata(
            dataset_id=dataset_id,
            tag=None,  # resolve whatever NEMAR holds, then compare below
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            metadata_timeout=metadata_timeout,
        )
    except (RuntimeError, ValueError) as exc:
        reason = str(exc).splitlines()[0]
        cprint(
            f"Could not get extra checksums from NEMAR ({reason}) "
            "Continuing with OpenNeuro's own verification."
        )
        return {}

    mirrored_tag = snapshot.id.removeprefix(f"{dataset_id}:")
    if mirrored_tag != tag:
        cprint(
            f"NEMAR mirrors revision {mirrored_tag} of {dataset_id}, but "
            f"revision {tag} is being downloaded, so its checksums do not "
            "apply. Continuing with OpenNeuro's own verification."
        )
        return {}

    return {
        f.filename: (f.checksum, f.checksum_algorithm)
        for f in snapshot.files
        if f.checksum is not None
        and f.checksum_algorithm is not None
        and f.filename not in _REWRITTEN_BY_NEMAR
    }
