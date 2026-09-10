"""Tests for downloading from the NEMAR mirror."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openneuro import _download, _nemar
from openneuro._download import (
    _apply_nemar_checksums,
    _get_local_tag,
    _make_hasher,
    _make_nemar_debug_hint,
)
from openneuro._models import DatasetFile, Snapshot
from openneuro._nemar import (
    _files_from_manifest,
    _NotFound,
    _openneuro_tag_from_metadata,
    get_metadata,
    to_nemar_id,
)

# -- dataset ID mapping --


@pytest.mark.parametrize(
    ("dataset_id", "expected"),
    [
        ("ds004840", "on004840"),
        ("ds000117", "on000117"),
        ("ds1", "on1"),
    ],
)
def test_to_nemar_id(dataset_id: str, expected: str):
    """OpenNeuro IDs map onto NEMAR IDs by swapping the prefix."""
    assert to_nemar_id(dataset_id) == expected


@pytest.mark.parametrize(
    "dataset_id",
    [
        "on004840",  # already a NEMAR ID
        "nm000103",  # NEMAR-native, no OpenNeuro counterpart
        "ds00abcd",  # not numeric
        "",
    ],
)
def test_to_nemar_id_rejects_non_openneuro_ids(dataset_id: str):
    """Anything that is not a ``ds<number>`` ID is refused."""
    with pytest.raises(ValueError, match="ds<number>"):
        to_nemar_id(dataset_id)


# -- provenance --


def test_openneuro_tag_from_metadata():
    """The OpenNeuro revision is read off the IsDerivedFrom DOI."""
    metadata = {
        "related_identifiers": [
            {
                "identifier": "https://github.com/nemarDatasets/on004840",
                "relation_type": "IsDescribedBy",
            },
            {
                "identifier": "10.18112/openneuro.ds004840",
                "relation_type": "IsVersionOf",
            },
            {
                "identifier": "10.18112/openneuro.ds004840.v1.0.1",
                "relation_type": "IsDerivedFrom",
            },
        ]
    }
    assert _openneuro_tag_from_metadata(metadata) == "1.0.1"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"related_identifiers": None},
        {"related_identifiers": []},
        # Present, but not an OpenNeuro DOI.
        {
            "related_identifiers": [
                {
                    "identifier": "10.82901/nemar.on004840",
                    "relation_type": "IsDerivedFrom",
                }
            ]
        },
        # Right DOI, wrong relation.
        {
            "related_identifiers": [
                {
                    "identifier": "10.18112/openneuro.ds004840.v1.0.1",
                    "relation_type": "IsVersionOf",
                }
            ]
        },
    ],
)
def test_openneuro_tag_from_metadata_missing(metadata):
    """Absent or unrecognised provenance reads as "unknown", not an error."""
    assert _openneuro_tag_from_metadata(metadata) is None


# -- manifest parsing --


def test_files_from_manifest():
    """Manifest entries become DatasetFiles, preferring the non-expiring URL."""
    manifest = [
        {
            "path": ".bidsignore",
            "size": 8,
            "checksum_algorithm": "git",
            "checksum": "66fda975",
            "bytes_url": "https://raw.githubusercontent.com/x/on1/v1/.bidsignore",
            "url": "https://raw.githubusercontent.com/x/on1/v1/.bidsignore",
        },
        {
            "path": "sub-01/eeg/sub-01_eeg.edf",
            "size": 22208868,
            "checksum_algorithm": "sha256",
            "checksum": "2837b4bd",
            "bytes_url": "https://data.nemar.org/on1/v1/sub-01/eeg/sub-01_eeg.edf",
            "url": "https://nemar.s3.amazonaws.com/...?X-Amz-Expires=3600",
        },
    ]
    files = _files_from_manifest(manifest, nemar_id="on1", version="v1")

    assert [f.filename for f in files] == [
        ".bidsignore",
        "sub-01/eeg/sub-01_eeg.edf",
    ]
    assert files[0].checksum_algorithm == "git"
    assert files[1].checksum == "2837b4bd"
    assert files[1].size == 22208868
    # `url` expires after an hour, so `bytes_url` must win.
    assert files[1].urls == ["https://data.nemar.org/on1/v1/sub-01/eeg/sub-01_eeg.edf"]


def test_files_from_manifest_falls_back_to_url():
    """`url` is used when the manifest carries no `bytes_url`."""
    files = _files_from_manifest(
        [{"path": "a.txt", "size": 1, "url": "https://example.com/a"}],
        nemar_id="on1",
        version="v1",
    )
    assert files[0].urls == ["https://example.com/a"]


def test_files_from_manifest_rejects_non_list():
    """A manifest that is not a list is a bug worth reporting, not a crash."""
    with pytest.raises(RuntimeError, match="not a list"):
        _files_from_manifest({"nope": True}, nemar_id="on1", version="v1")


# -- get_metadata, with the network stubbed out --


def _fake_get_json(responses: dict[str, object]):
    """Return a `_get_json` stub serving `responses`, keyed by URL suffix.

    A URL with no entry raises `_NotFound`, standing in for NEMAR's 404.
    """

    def _stub(url: str, **kwargs: object):
        for suffix, payload in responses.items():
            if url.endswith(suffix):
                return payload
        raise _NotFound(url)

    return _stub


_KWARGS = dict(max_retries=0, retry_backoff=0.0, metadata_timeout=1.0)

_MANIFEST = [
    {
        "path": "dataset_description.json",
        "size": 3,
        "checksum_algorithm": "sha256",
        "checksum": "abc",
        "bytes_url": "https://data.nemar.org/on004840/v1.0.0/dataset_description.json",
    }
]
_METADATA = {
    "related_identifiers": [
        {
            "identifier": "10.18112/openneuro.ds004840.v1.0.1",
            "relation_type": "IsDerivedFrom",
        }
    ]
}
_VERSIONS = {"dataset_id": "on004840", "latest": "v1.0.0"}


def test_get_metadata_reports_the_openneuro_tag():
    """The snapshot ID carries the *OpenNeuro* tag, not NEMAR's version."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": _METADATA,
        "/manifest.json": _MANIFEST,
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        snapshot = get_metadata(dataset_id="ds004840", tag=None, **_KWARGS)

    # NEMAR calls this v1.0.0; OpenNeuro calls it 1.0.1. We report the latter.
    assert snapshot.id == "ds004840:1.0.1"
    assert snapshot.files[0].checksum_algorithm == "sha256"


def test_get_metadata_accepts_the_matching_tag():
    """Requesting the OpenNeuro revision NEMAR holds succeeds."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": _METADATA,
        "/manifest.json": _MANIFEST,
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        snapshot = get_metadata(dataset_id="ds004840", tag="1.0.1", **_KWARGS)
    assert snapshot.id == "ds004840:1.0.1"


def test_get_metadata_rejects_a_different_tag():
    """A revision NEMAR does not hold fails with an actionable message."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": _METADATA,
        "/manifest.json": _MANIFEST,
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        with pytest.raises(RuntimeError) as exc_info:
            get_metadata(dataset_id="ds004840", tag="1.0.0", **_KWARGS)

    message = str(exc_info.value)
    # It must say what NEMAR *does* have, and how to get what was asked for.
    assert "1.0.1" in message
    assert 'source="openneuro"' in message


def test_get_metadata_dataset_not_mirrored():
    """A dataset NEMAR does not carry names OpenNeuro as the way out."""
    with patch.object(_nemar, "_get_json", _fake_get_json({})):
        with pytest.raises(RuntimeError) as exc_info:
            get_metadata(dataset_id="ds000001", tag=None, **_KWARGS)

    message = str(exc_info.value)
    assert "ds000001 is not available from NEMAR" in message
    assert 'source="openneuro"' in message


def test_get_metadata_no_published_version():
    """A listed dataset with no published version is reported clearly."""
    responses = {"/on005691/": {"dataset_id": "on005691", "latest": None}}
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        with pytest.raises(RuntimeError, match="has not published"):
            get_metadata(dataset_id="ds005691", tag=None, **_KWARGS)


def test_get_metadata_unknown_provenance_without_tag(capsys):
    """Missing provenance is survivable when no revision was requested."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": {},
        "/manifest.json": _MANIFEST,
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        snapshot = get_metadata(dataset_id="ds004840", tag=None, **_KWARGS)
    # Falls back to NEMAR's own version, and says so.
    assert snapshot.id == "ds004840:v1.0.0"


def test_get_metadata_unknown_provenance_with_tag():
    """Missing provenance is fatal when a specific revision was requested."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": {},
        "/manifest.json": _MANIFEST,
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        with pytest.raises(RuntimeError, match="does not record"):
            get_metadata(dataset_id="ds004840", tag="1.0.1", **_KWARGS)


def test_get_metadata_empty_manifest():
    """An empty manifest is an error rather than a silent no-op download."""
    responses = {
        "/on004840/": _VERSIONS,
        "/metadata.json": _METADATA,
        "/manifest.json": [],
    }
    with patch.object(_nemar, "_get_json", _fake_get_json(responses)):
        with pytest.raises(RuntimeError, match="lists no files"):
            get_metadata(dataset_id="ds004840", tag=None, **_KWARGS)


# -- _get_json --


def _mock_session(status_code: int, json_data: object = None, json_error=None):
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = json_data

    client = MagicMock()
    client.get.return_value = response
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


def test_get_json_raises_not_found_on_404():
    """HTTP 404 becomes `_NotFound` so callers can explain it in context."""
    with patch.object(_nemar.niquests, "Session", return_value=_mock_session(404)):
        with pytest.raises(_NotFound):
            _nemar._get_json(
                "https://data.nemar.org/on1/",
                what="testing",
                timeout=1.0,
                max_retries=0,
                retry_backoff=0.0,
            )


def test_get_json_raises_on_other_errors():
    """A non-retryable error status is surfaced with its code."""
    with patch.object(_nemar.niquests, "Session", return_value=_mock_session(403)):
        with pytest.raises(RuntimeError, match="HTTP 403"):
            _nemar._get_json(
                "https://data.nemar.org/on1/",
                what="testing",
                timeout=1.0,
                max_retries=0,
                retry_backoff=0.0,
            )


def test_get_json_retries_then_times_out():
    """Retryable statuses are retried, and exhaustion reports a timeout."""
    client = _mock_session(503)
    with patch.object(_nemar.niquests, "Session", return_value=client):
        with pytest.raises(RuntimeError, match="Timeout when testing"):
            _nemar._get_json(
                "https://data.nemar.org/on1/",
                what="testing",
                timeout=1.0,
                max_retries=2,
                retry_backoff=0.0,
            )
    assert client.get.call_count == 3  # the initial attempt plus two retries


def test_get_json_raises_on_non_json():
    """A non-JSON body is reported rather than propagating a decode error."""
    client = _mock_session(200, json_error=json.JSONDecodeError("", "", 0))
    with patch.object(_nemar.niquests, "Session", return_value=client):
        with pytest.raises(RuntimeError, match="non-JSON"):
            _nemar._get_json(
                "https://data.nemar.org/on1/",
                what="testing",
                timeout=1.0,
                max_retries=0,
                retry_backoff=0.0,
            )


# -- checksum algorithms --


def test_make_hasher_git_blob():
    """The "git" algorithm is a git blob hash, not a plain SHA-1."""
    content = b"hello world\n"
    hasher = _make_hasher("git", size=len(content))
    hasher.update(content)
    expected = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()

    assert hasher.hexdigest() == expected
    assert hasher.hexdigest() != hashlib.sha1(content).hexdigest()


def test_make_hasher_git_blob_requires_size():
    """Without the size the length prefix cannot be built."""
    with pytest.raises(ValueError, match="requires the file size"):
        _make_hasher("git", size=None)


@pytest.mark.parametrize(
    ("algorithm", "reference"),
    [("md5", hashlib.md5), ("sha256", hashlib.sha256)],
)
def test_make_hasher_plain_algorithms(algorithm, reference):
    """md5 and sha256 hash the contents directly."""
    content = b"some bytes"
    hasher = _make_hasher(algorithm, size=len(content))
    hasher.update(content)
    assert hasher.hexdigest() == reference(content).hexdigest()


def test_make_hasher_rejects_unknown():
    """An unrecognised algorithm is refused rather than silently ignored."""
    with pytest.raises(ValueError, match="Unknown checksum algorithm"):
        _make_hasher("crc32", size=1)  # type: ignore[arg-type]


# -- local revision detection across sources --


def _write_description(path: Path, content: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset_description.json").write_text(json.dumps(content), "utf-8")


def test_get_local_tag_from_nemar_directory(tmp_path: Path):
    """NEMAR rewrites DatasetDOI, but records the source revision separately."""
    _write_description(
        tmp_path,
        {
            "DatasetDOI": "10.82901/nemar.on004840",
            "SourceDatasets": [{"DOI": "doi:10.18112/openneuro.ds004840.v1.0.1"}],
            "Version": "1.0.0",
        },
    )
    assert _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path) == "1.0.1"


def test_get_local_tag_from_nemar_directory_without_provenance(tmp_path: Path):
    """A NEMAR DOI alone says nothing about the OpenNeuro revision."""
    _write_description(tmp_path, {"DatasetDOI": "10.82901/nemar.on004840"})
    assert _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path) is None


def test_get_local_tag_rejects_a_different_dataset(tmp_path: Path):
    """A NEMAR DOI for another dataset is still caught as a mismatch."""
    _write_description(tmp_path, {"DatasetDOI": "10.82901/nemar.on000117"})
    with pytest.raises(RuntimeError, match="appears to be different"):
        _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path)


def test_get_local_tag_survives_a_corrupt_description(tmp_path: Path):
    """A damaged dataset_description.json is replaceable, not fatal.

    Returning `None` lets the download re-fetch the file; raising a decode
    error would leave the user with a directory they cannot repair.
    """
    (tmp_path / "dataset_description.json").write_bytes(b"\x00\x00\x00")
    assert _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path) is None

    (tmp_path / "dataset_description.json").write_text("not json", "utf-8")
    assert _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path) is None

    # A valid JSON document that is not an object is equally unusable.
    (tmp_path / "dataset_description.json").write_text("[1, 2]", "utf-8")
    assert _get_local_tag(dataset_id="ds004840", dataset_dir=tmp_path) is None


# -- cross-source verification: NEMAR checksums for an OpenNeuro download --


_CROSS_MANIFEST = [
    {
        "path": "sub-01/meg/big.meg4",
        "size": 1175040008,
        "checksum_algorithm": "md5",
        "checksum": "9f8b091f26abb85986ebe68ce48661f3",
        "bytes_url": "https://data.nemar.org/on000246/v1/sub-01/meg/big.meg4",
    },
    {
        # NEMAR rewrites this one, so its checksum must never be handed out.
        "path": "dataset_description.json",
        "size": 996,
        "checksum_algorithm": "git",
        "checksum": "deadbeef",
        "bytes_url": "https://data.nemar.org/on000246/v1/dataset_description.json",
    },
]


def _cross_responses(metadata=_METADATA):
    return {
        "/on000246/": {"dataset_id": "on000246", "latest": "v1.0.0"},
        "/metadata.json": metadata,
        "/manifest.json": _CROSS_MANIFEST,
    }


def test_get_checksums_excludes_rewritten_files():
    """NEMAR's own dataset_description.json must never verify OpenNeuro's."""
    with patch.object(_nemar, "_get_json", _fake_get_json(_cross_responses())):
        checksums = _nemar.get_checksums(dataset_id="ds000246", tag="1.0.1", **_KWARGS)

    assert checksums == {
        "sub-01/meg/big.meg4": ("9f8b091f26abb85986ebe68ce48661f3", "md5")
    }
    assert "dataset_description.json" not in checksums


def test_get_checksums_refuses_a_different_revision(capsys):
    """A mirror that has fallen behind describes different bytes."""
    with patch.object(_nemar, "_get_json", _fake_get_json(_cross_responses())):
        # NEMAR mirrors 1.0.1 (per _METADATA); we are downloading 1.0.0.
        checksums = _nemar.get_checksums(dataset_id="ds000246", tag="1.0.0", **_KWARGS)

    assert checksums == {}
    assert "do not apply" in capsys.readouterr().err


def test_get_checksums_degrades_when_nemar_is_unavailable(capsys):
    """NEMAR being down must not break a working OpenNeuro download."""
    with patch.object(_nemar, "_get_json", _fake_get_json({})):  # everything 404s
        checksums = _nemar.get_checksums(dataset_id="ds000246", tag="1.0.1", **_KWARGS)

    assert checksums == {}
    assert "Continuing with OpenNeuro" in capsys.readouterr().err


def test_apply_nemar_checksums_merges_only_matching_files():
    """Files NEMAR covers gain a checksum; the rest are untouched."""
    files = [
        DatasetFile(
            filename="sub-01/meg/big.meg4", urls=["https://on/big"], size=1, id="a"
        ),
        DatasetFile(filename="README", urls=["https://on/readme"], size=2, id="b"),
    ]
    with patch.object(_nemar, "_get_json", _fake_get_json(_cross_responses())):
        merged = _apply_nemar_checksums(
            files,
            dataset_id="ds000246",
            tag="1.0.1",
            max_retries=0,
            retry_backoff=0.0,
            metadata_timeout=1.0,
        )

    by_name = {f.filename: f for f in merged}
    assert by_name["sub-01/meg/big.meg4"].checksum_algorithm == "md5"
    # NEMAR stores the README as README.md, so this one has no counterpart.
    assert by_name["README"].checksum is None
    # The originals are left alone rather than mutated in place.
    assert files[0].checksum is None


def test_apply_nemar_checksums_returns_input_when_unavailable():
    """An unreachable mirror leaves the file list exactly as it was."""
    files = [DatasetFile(filename="a.txt", urls=["https://on/a"], size=1, id="a")]
    with patch.object(_nemar, "_get_json", _fake_get_json({})):
        merged = _apply_nemar_checksums(
            files,
            dataset_id="ds000246",
            tag="1.0.1",
            max_retries=0,
            retry_backoff=0.0,
            metadata_timeout=1.0,
        )
    assert merged == files


# -- failure hints --


def test_nemar_debug_hint_uses_resolvable_coordinates():
    """The hint must not send readers to a URL that 404s.

    NEMAR addresses the dataset by its own ID and its own version numbering, so
    a hint containing `<dataset>`/`<version>` placeholders invites the reader to
    substitute the OpenNeuro ones and land on a 404.
    """
    hint = _make_nemar_debug_hint("ds004840")

    assert "https://data.nemar.org/on004840/" in hint
    # Never the OpenNeuro ID as a NEMAR path segment...
    assert "data.nemar.org/ds004840" not in hint
    # ...and nothing left for the reader to fill in.
    assert "<dataset>" not in hint
    assert "<version>" not in hint
    # The OpenNeuro ID still appears, to explain the mapping.
    assert "ds004840" in hint
    assert 'source="openneuro"' in hint


def test_nemar_debug_hint_reaches_failures(tmp_path: Path):
    """A failing NEMAR download surfaces the NEMAR hint, not the GraphQL one."""
    snapshot = Snapshot(
        id="ds004840:1.0.1",
        files=[DatasetFile(filename="a.bin", urls=None, size=1, id="a")],
    )
    with (
        patch.object(_download, "_get_download_metadata", return_value=snapshot),
        patch.object(_download, "_get_local_tag", return_value=None),
        pytest.raises(RuntimeError, match="Failed to download"),
    ):
        _download.download(
            dataset="ds004840", tag="1.0.1", target_dir=tmp_path, source="nemar"
        )
