"""Test downloading and authentication."""

import asyncio
import copy
import io
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import niquests
import pytest
from rich.progress import Progress, TaskID

import openneuro
import openneuro._config
from openneuro import _download
from openneuro._download import (
    _download_file,
    _format_size,
    _retrieve_and_write_to_disk,
    download,
)
from openneuro._models import DatasetFile, Snapshot
from tests.utils import load_json

dataset_id_aws = "ds000246"
tag_aws = "1.0.0"
include_aws = "sub-0001/anat"
exclude_aws: list[str] = []

dataset_id_on = "ds000117"
tag_on = None
include_on = "sub-16/ses-meg"
exclude_on = "*.fif"  # save GBs of downloads

invalid_tag = "abcdefg"


@pytest.mark.parametrize(
    ("dataset_id", "tag", "include", "exclude"),
    [
        # errors on this one as of 2026/01/19
        pytest.param(
            dataset_id_aws,
            tag_aws,
            include_aws,
            exclude_aws,
            id="aws-ds000246",
            marks=pytest.mark.flaky(reruns=5, reruns_delay=5),
        ),
        pytest.param(dataset_id_on, tag_on, include_on, exclude_on, id="on-ds000117"),
    ],
)
def test_download(tmp_path: Path, dataset_id, tag, include, exclude):
    """Test downloading some files."""
    download(
        dataset=dataset_id,
        tag=tag,
        target_dir=tmp_path,
        include=include,
        exclude=exclude,
    )


def test_download_invalid_tag(
    tmp_path: Path, dataset_id=dataset_id_on, invalid_tag=invalid_tag
):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match="snapshot.*does not exist"):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)


@pytest.mark.flaky(reruns=5, reruns_delay=5)
def test_resume_download(tmp_path: Path):
    """Test resuming of a dataset download."""
    dataset = "ds000246"
    tag = "1.0.1"
    include = ["CHANGES"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)

    # Download some more files
    include = ["sub-0001/meg/*.jpg"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)

    # Download from a different revision / tag
    new_tag = "00001"
    include = ["CHANGES"]
    with pytest.raises(FileExistsError, match=f"revision {tag} exists"):
        download(dataset=dataset, tag=new_tag, target_dir=tmp_path, include=include)

    # Try to "resume" from a different dataset
    new_dataset = "ds000117"
    with pytest.raises(RuntimeError, match="existing dataset.*appears to be different"):
        download(dataset=new_dataset, target_dir=tmp_path, include=include)

    # Remove "DatasetDOI" from JSON
    json_path = tmp_path / "dataset_description.json"
    with json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    del dataset_json["DatasetDOI"]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f)

    with pytest.raises(RuntimeError, match=r'does not contain "DatasetDOI"'):
        download(dataset=dataset, target_dir=tmp_path)

    # We should be able to resume a download even if "dataset_description.json"
    # is missing
    json_path.unlink()
    include = ["sub-0001/meg/sub-0001_coordsystem.json"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)


def test_ds000248(tmp_path: Path):
    """Test a dataset for that we ship default excludes."""
    dataset = "ds000248"
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)


def test_doi_handling(tmp_path: Path):
    """Test that we can handle DOIs that start with 'doi:`."""
    dataset = "ds000248"
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)

    # Now inject a `doi:` prefix into the DOI
    dataset_description_path = tmp_path / "dataset_description.json"
    dataset_description_text = dataset_description_path.read_text(encoding="utf-8")
    dataset_description = json.loads(dataset_description_text)
    # Make sure we can dumps to get the same thing back (if they change their
    # indent 4->8 for example, we might try to resume our download of the file
    # and things will break in a challenging way)
    dataset_description_rt = json.dumps(dataset_description, indent=4)
    assert dataset_description_text == dataset_description_rt
    # Ensure the dataset doesn't already have the problematic prefix, then add
    assert not dataset_description["DatasetDOI"].startswith("doi:")
    dataset_description["DatasetDOI"] = "doi:" + dataset_description["DatasetDOI"]
    dataset_description_path.write_text(
        data=json.dumps(dataset_description, indent=4), encoding="utf-8"
    )

    # Try to download again
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)


def test_restricted_dataset(tmp_path: Path, openneuro_token: str):
    """Test downloading a restricted dataset."""
    with patch.object(openneuro._config, "CONFIG_PATH", tmp_path / ".openneuro"):
        with patch("getpass.getpass", lambda _: openneuro_token):
            openneuro._config.init_config()

        # This is a restricted dataset that is only available if the API token
        # was used correctly.
        download(dataset="ds006412", include="README.txt", target_dir=tmp_path)

    assert (tmp_path / "README.txt").exists()


@pytest.mark.parametrize(
    ("dataset", "include", "expected_files"),
    load_json("expected_files_test_cases.json"),
)
def test_download_file_list_generation(
    dataset: str, include: list[str], expected_files: list[str], tmp_path: Path
):
    """Test that download generates the correct list of files.

    This test verifies the file filtering logic by mocking the
    metadata retrieval and checking that the correct files are
    selected based on include/exclude patterns.

    Test cases are loaded from `expected_files_test_cases.json`
    which contains an array of test case tuples. Each tuple has
    the structure:
    [dataset_id, include_patterns, expected_file_list]

    Where:
    - dataset_id: OpenNeuro dataset identifier (e.g., "ds000117")
    - include_patterns: List of glob patterns to include files
      (e.g., ["*.tsv"], ["sub-01"], ["sub-01/**"])
    - expected_file_list: Complete list of files that should be
      selected, including dataset metadata files

    The test uses `mock_metadata_ds000117.json` which contains
    mock OpenNeuro metadata for dataset ds000117. This file
    simulates the API response with file listings including
    filenames, URLs, and sizes for realistic testing without
    requiring actual API calls. Having a mock
    metadata makes it easy to control which files should be
    selected with different include patterns. The `mock_metadata_ds000117.json`
    file was built manually using the following directory structure:

    |ds000117/
    |--- CHANGES
    |--- README
    |--- dataset_description.json
    |--- participants.json
    |--- participants.tsv
    |--- derivatives/
    |------ freesurfer/
    |--------- sub-01/
    |------------ ses-mri/
    |--------------- anat/
    |------------------ label/
    |--------------------- .lh.BA.thresh.annot.f3h5wZ
    |--------------------- lh.BA.annot
    |--------------------- lh.BA.thresh.annot
    |--------------------- lh.aparc.DKTatlas40.annot
    |--------------------- lh.aparc.a2009s.annot
    |--------------------- lh.aparc.annot
    |--------------------- rh.BA.annot
    |--------------------- rh.BA.thresh.annot
    |--------------------- rh.aparc.DKTatlas40.annot
    |--------------------- rh.aparc.a2009s.annot
    |--------------------- rh.aparc.annot
    |------------------ mri/
    |--------------------- T1.mgz
    |--------------------- aseg.mgz
    |------------------ surf/
    |--------------------- lh.pial
    |--------------------- lh.sphere.reg
    |--------------------- lh.white
    |--------------------- rh.pial
    |--------------------- rh.sphere.reg
    |--------------------- rh.white
    |--------- sub-02/
    |------------ ses-mri/
    |--------------- anat/
    |------------------ label/
    |--------------------- lh.BA.annot
    |--------------------- lh.BA.thresh.annot
    |--------------------- lh.aparc.DKTatlas40.annot
    |--------------------- lh.aparc.a2009s.annot
    |--------------------- lh.aparc.annot
    |--------------------- rh.BA.annot
    |--------------------- rh.BA.thresh.annot
    |--------------------- rh.aparc.DKTatlas40.annot
    |--------------------- rh.aparc.a2009s.annot
    |--------------------- rh.aparc.annot
    |------------------ mri/
    |--------------------- T1.mgz
    |--------------------- aseg.mgz
    |------------------ surf/
    |--------------------- lh.pial
    |--------------------- lh.sphere.reg
    |--------------------- lh.white
    |--------------------- rh.pial
    |--------------------- rh.sphere.reg
    |--------------------- rh.white
    |--- sub-01/
    |------ ses-meg/
    |--------- sub-01_ses-meg_scans.tsv
    |--------- sub-01_ses-meg_task-facerecognition_channels.tsv
    |--------- sub-01_ses-meg_task-facerecognition_meg.json
    |--------- beh/
    |------------ sub-01_ses-meg_task-facerecognition_events.tsv
    |--------- meg/
    |------------ sub-01_ses-meg_coordsystem.json
    |------------ sub-01_ses-meg_headshape.pos
    |------------ sub-01_ses-meg_task-facerecognition_run-01_events.tsv
    |------------ sub-01_ses-meg_task-facerecognition_run-01_meg.fif
    |------------ sub-01_ses-meg_task-facerecognition_run-02_events.tsv
    |------------ sub-01_ses-meg_task-facerecognition_run-02_meg.fif
    |------ ses-mri/
    |--------- anat/
    |------------ sub-01_ses-mri_acq-mprage_T1w.json
    |------------ sub-01_ses-mri_acq-mprage_T1w.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz
    |------------ sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz
    |--------- dwi/
    |------------ sub-01_ses-mri_dwi.bval
    |------------ sub-01_ses-mri_dwi.bvec
    |------------ sub-01_ses-mri_dwi.json
    |------------ sub-01_ses-mri_dwi.nii.gz
    |--------- fmap/
    |------------ sub-01_ses-mri_magnitude1.json
    |------------ sub-01_ses-mri_magnitude1.nii
    |------------ sub-01_ses-mri_magnitude2.json
    |------------ sub-01_ses-mri_magnitude2.nii
    |------------ sub-01_ses-mri_phasediff.json
    |------------ sub-01_ses-mri_phasediff.nii
    |--------- func/
    |------------ sub-01_ses-mri_task-facerecognition_run-01_bold.json
    |------------ sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz
    |------------ sub-01_ses-mri_task-facerecognition_run-01_events.tsv
    |------------ sub-01_ses-mri_task-facerecognition_run-02_bold.json
    |------------ sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz
    |------------ sub-01_ses-mri_task-facerecognition_run-02_events.tsv
    |--- sub-02/
    |------ ses-meg/
    |--------- sub-02_ses-meg_scans.tsv
    |--------- sub-02_ses-meg_task-facerecognition_channels.tsv
    |--------- sub-02_ses-meg_task-facerecognition_meg.json
    |--------- beh/
    |------------ sub-02_ses-meg_task-facerecognition_events.tsv
    |--------- meg/
    |------------ sub-02_ses-meg_coordsystem.json
    |------------ sub-02_ses-meg_headshape.pos
    |------------ sub-02_ses-meg_task-facerecognition_run-01_events.tsv
    |------------ sub-02_ses-meg_task-facerecognition_run-01_meg.fif
    |------------ sub-02_ses-meg_task-facerecognition_run-02_events.tsv
    |------------ sub-02_ses-meg_task-facerecognition_run-02_meg.fif
    |------ ses-mri/
    |--------- anat/
    |------------ sub-02_ses-mri_acq-mprage_T1w.json
    |------------ sub-02_ses-mri_acq-mprage_T1w.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-1_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-2_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-3_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-4_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-5_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-6_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-1_echo-7_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-1_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-2_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-3_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-4_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-5_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-6_FLASH.nii.gz
    |------------ sub-02_ses-mri_run-2_echo-7_FLASH.nii.gz
    |--------- dwi/
    |------------ sub-02_ses-mri_dwi.bval
    |------------ sub-02_ses-mri_dwi.bvec
    |------------ sub-02_ses-mri_dwi.json
    |------------ sub-02_ses-mri_dwi.nii.gz
    |--------- fmap/
    |------------ sub-02_ses-mri_magnitude1.json
    |------------ sub-02_ses-mri_magnitude1.nii
    |------------ sub-02_ses-mri_magnitude2.json
    |------------ sub-02_ses-mri_magnitude2.nii
    |------------ sub-02_ses-mri_phasediff.json
    |------------ sub-02_ses-mri_phasediff.nii
    |--------- func/
    |------------ sub-02_ses-mri_task-facerecognition_run-01_bold.json
    |------------ sub-02_ses-mri_task-facerecognition_run-01_bold.nii.gz
    |------------ sub-02_ses-mri_task-facerecognition_run-01_events.tsv
    |------------ sub-02_ses-mri_task-facerecognition_run-02_bold.json
    |------------ sub-02_ses-mri_task-facerecognition_run-02_bold.nii.gz
    |------------ sub-02_ses-mri_task-facerecognition_run-02_events.tsv
    |--- sub-emptyroom/
    |------ ses-20090409/
    |--------- sub-emptyroom_ses-20090409_scans.tsv
    |--------- meg/
    |------------ sub-emptyroom_ses-20090409_task-noise_meg.fif

    To add more test cases:
    1. Open `src/openneuro/tests/data/expected_files_test_cases.json`
    2. Add new test case as: ["dataset", ["pattern1", "pattern2"],
      ["file1", "file2", ...]]
    3. Include dataset metadata files (CHANGES, README, etc.)
    4. Ensure all expected files match the include patterns
    5. Validate JSON syntax and file paths are correct
    """
    MOCK_METADATA = Snapshot.model_validate(load_json(f"mock_metadata_{dataset}.json"))

    def mock_get_download_metadata(*args, **kwargs):
        return copy.deepcopy(MOCK_METADATA)

    def mock_get_local_tag(*args, **kwargs):
        return None

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments."""
        return None

    with (
        patch.object(
            _download, "_get_download_metadata", side_effect=mock_get_download_metadata
        ) as mock_get_download_metadata,
        patch.object(
            _download, "_get_local_tag", side_effect=mock_get_local_tag
        ) as mock_get_local_tag,
        patch.object(
            _download, "_download_files", side_effect=_download_files_spy
        ) as _download_files_spy,
    ):
        # Run the function with an include pattern
        _download.download(
            dataset=dataset,
            target_dir=Path(tmp_path),
            include=include,
        )

        files_arg = _download_files_spy.call_args[1]["files"]
        files_arg = [file.filename for file in files_arg]
        assert len(files_arg) == len(expected_files), (
            f"Expected {len(expected_files)} files, got {len(files_arg)}"
        )
        for file in files_arg:
            assert file in expected_files, f"File {file} not found in expected files"


@pytest.mark.parametrize(
    ("dataset", "include", "expected_num_files"),
    load_json("expected_file_count_test_cases.json"),
)
def test_download_file_count(
    dataset: str, include: list[str], expected_num_files: int, tmp_path: Path
):
    """Test that download generates the correct number of files.

    This test verifies the file filtering logic by mocking
    the metadata retrieval and checking that the correct
    number of files are selected based on include patterns.

    Test cases are loaded from `expected_file_count_test_cases.json`
    which contains an array of test case tuples. Each tuple has
    the structure:
    [dataset_id, include_patterns, expected_file_count]

    Where:
    - dataset_id: OpenNeuro dataset identifier (e.g., "ds000117")
    - include_patterns: List of glob patterns to include files
      (e.g., ["*"], ["sub-01"], ["sub-01/**/*.tsv"])
    - expected_file_count: Integer count of files that should
      be selected by the include patterns

    To add more test cases:
    1. Open `src/openneuro/tests/data/expected_file_count_test_cases.json`
    2. Add new test case as: ["dataset", ["pattern1", "pattern2"],
      count_number]
    3. Count should include dataset metadata files in total
    4. Verify count matches actual files selected by patterns
    5. Ensure JSON syntax is valid and numbers are integers

    """
    MOCK_METADATA = Snapshot.model_validate(load_json(f"mock_metadata_{dataset}.json"))

    def mock_get_download_metadata(*args, **kwargs):
        return copy.deepcopy(MOCK_METADATA)

    def mock_get_local_tag(*args, **kwargs):
        return None

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments."""
        return None

    with (
        patch.object(
            _download,
            "_get_download_metadata",
            side_effect=mock_get_download_metadata,
        ),
        patch.object(_download, "_get_local_tag", side_effect=mock_get_local_tag),
        patch.object(
            _download, "_download_files", side_effect=_download_files_spy
        ) as _download_files_spy,
    ):
        # Run the function with an include pattern
        _download.download(
            dataset=dataset,
            tag="1.1.0",
            target_dir=tmp_path,
            include=include,
        )

        files_arg = _download_files_spy.call_args[1]["files"]
        files_arg = [file.filename for file in files_arg]
        assert len(files_arg) == expected_num_files, (
            f"Expected {expected_num_files} files, got {len(files_arg)}"
        )


def test_bidsignore_always_downloaded(tmp_path: Path):
    """`.bidsignore` is downloaded despite being a dotfile (gh-327)."""
    names = ["dataset_description.json", ".bidsignore", "sub-01/anat/T1w.nii.gz"]
    snapshot = Snapshot(
        id="ds000000:1.0.0",
        files=[DatasetFile(filename=n, urls=["http://x"], size=1, id=n) for n in names],
    )

    async def _spy(*, files, **kwargs):
        return None

    def selected(**kwargs) -> set[str]:
        with (
            patch.object(_download, "_get_download_metadata", return_value=snapshot),
            patch.object(_download, "_get_local_tag", return_value=None),
            patch.object(_download, "_download_files", side_effect=_spy) as spy,
        ):
            download(dataset="ds000000", target_dir=tmp_path / "out", **kwargs)
        return {f.filename for f in spy.call_args[1]["files"]}

    # Dotfiles are skipped by default, but `.bidsignore` is essential and is
    # kept alongside the normal (non-dotfile) files.
    assert selected() == set(names)
    # An unrelated `include` still pulls in `.bidsignore`...
    assert ".bidsignore" in selected(include=["sub-01"])
    # ...as does an `exclude` pattern that would otherwise drop it.
    assert ".bidsignore" in selected(exclude=["**/.*"])


def _make_dataset_client(
    *,
    bodies: dict[str, bytes],
    get_status: dict[str, int] | None = None,
    etag: str | None = None,
):
    """Fake `niquests.AsyncSession` serving a whole dataset, keyed by filename.

    `bodies` maps a filename to the bytes its GET yields; `get_status` overrides
    the GET status code for a filename; `etag` is reported by every HEAD.
    """
    get_status = get_status or {}

    def _key(url: str) -> str:
        return next((name for name in bodies if url.endswith(name)), "")

    async def head(url, *, headers=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.headers = {"etag": f'"{etag}"'} if etag else {}
        return resp

    class _FakeStream:
        def __init__(self, body: bytes, status: int):
            self.status_code = status
            self.ok = status < 400
            self._body = body

        async def iter_content(self, chunk_size=-1, decode_unicode=False):
            body = self._body

            async def gen():
                yield body

            return gen()

        async def close(self):
            pass

    async def get(url, *, headers=None, timeout=None, stream=None):
        name = _key(url)
        return _FakeStream(bodies.get(name, b""), get_status.get(name, 200))

    client = AsyncMock()
    client.head = head
    client.get = get
    return client


def _run_download_files(tmp_path: Path, client, files, **kwargs):
    """Drive the real `_download_files` against a fake session."""

    class _Session:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *exc_info):
            return False

    stats = _download._DownloadStats()
    with patch.object(_download.niquests, "AsyncSession", lambda **kw: _Session()):
        failures = asyncio.run(
            _download._download_files(
                target_dir=tmp_path,
                files=files,
                verify_hash=kwargs.pop("verify_hash", False),
                verify_size=True,
                max_retries=kwargs.pop("max_retries", 1),
                retry_backoff=0.0,
                max_concurrent_downloads=3,
                query_str="query {}",
                stats=stats,
                **kwargs,
            )
        )
    return failures, stats


def test_download_files_collects_failures(tmp_path: Path):
    """Terminal failures are collected while the other files still finish.

    Exercises the real `_download_files` machinery (gh-309), unlike
    `test_partial_download_failure`, which stubs out `_download_file`.
    """
    names = ["ok1.bin", "gone.bin", "short.bin", "ok2.bin", "nourl.bin"]
    files = [
        DatasetFile(
            filename=n,
            # `nourl.bin` has no URLs at all, so it fails before any request.
            urls=None if n == "nourl.bin" else [f"https://example.com/{n}"],
            size=100,
            id=n,
        )
        for n in names
    ]
    failures, stats = _run_download_files(
        tmp_path,
        _make_dataset_client(
            bodies={n: b"x" * 100 for n in names} | {"short.bin": b"x" * 150},
            get_status={"gone.bin": 404},
        ),
        files,
    )

    reasons = {path: exc.reason for path, exc in failures}
    assert set(reasons) == {"gone.bin", "short.bin", "nourl.bin"}
    assert "HTTP 404" in reasons["gone.bin"]
    assert "Size mismatch" in reasons["short.bin"]
    assert "No download URLs" in reasons["nourl.bin"]

    # The healthy files completed despite their neighbours failing.
    for name in ("ok1.bin", "ok2.bin"):
        assert (tmp_path / name).read_bytes() == b"x" * 100
    assert stats.n_files == 2
    assert stats.n_bytes == 200


def test_overall_progress_not_double_counted_on_retry(tmp_path: Path):
    """Discarding a bad file on retry must uncount the bytes it contributed.

    A hash mismatch is retried (gh-309), and each attempt re-downloads the whole
    file; without a rollback the overall bar counted every attempt.
    """
    name = "bad.bin"
    files = [
        DatasetFile(
            filename=name, urls=[f"https://example.com/{name}"], size=100, id=name
        )
    ]
    max_retries = 2
    seen: list[float] = []

    class _RecordingProgress(Progress):
        def update(self, task_id, **kwargs):
            super().update(task_id, **kwargs)
            for task in self.tasks:
                if task.id == task_id and task.description == "Overall":
                    seen.append(task.completed)

    with patch.object(
        _download, "_make_progress", lambda: _RecordingProgress(disable=True)
    ):
        failures, _ = _run_download_files(
            tmp_path,
            # An etag that cannot match the body, so every attempt mismatches.
            _make_dataset_client(bodies={name: b"x" * 100}, etag="0" * 32),
            files,
            verify_hash=True,
            max_retries=max_retries,
        )

    assert len(failures) == 1
    assert "Hash mismatch" in failures[0][1].reason
    assert f"{max_retries} retries" in failures[0][1].reason
    # Every attempt streams the full file, but the bar must never run past it.
    assert max(seen) == 100


def test_partial_download_failure(tmp_path: Path) -> None:
    """A single file failure must not abort other downloads."""
    metadata = Snapshot.model_validate(load_json("mock_metadata_ds000117.json"))
    fail_filename = "participants.tsv"
    attempted: list[str] = []

    async def patched_download_file(*, remote_path: str, **kwargs):
        attempted.append(remote_path)
        if remote_path == fail_filename:
            raise _download._DownloadError(reason="Size mismatch.", hint="")

    with (
        patch.object(_download, "_get_download_metadata", return_value=metadata),
        patch.object(_download, "_get_local_tag", return_value=None),
        patch.object(_download, "_download_file", side_effect=patched_download_file),
    ):
        with pytest.raises(RuntimeError, match="Failed to download 1 file"):
            download(dataset="ds000117", tag="1.1.0", target_dir=tmp_path)

    assert fail_filename in attempted
    assert len(attempted) > 1


# -- Glob matching tests --


@pytest.mark.parametrize(
    ("filenames", "patterns", "expected"),
    [
        # Leading / anchors to root
        (
            ["participants.tsv", "README", "sub-01/ses-meg/file.tsv"],
            ["/*.tsv"],
            {"/*.tsv": {"participants.tsv"}},
        ),
        # * does not cross /
        (
            ["sub-01/file.tsv", "sub-01/ses-meg/file.tsv"],
            ["sub-01/*.tsv"],
            {"sub-01/*.tsv": {"sub-01/file.tsv"}},
        ),
        # ** crosses /
        (
            ["sub-01/ses-meg/file.tsv", "sub-01/a/b/c/file.tsv", "sub-02/file.tsv"],
            ["sub-01/**/*.tsv"],
            {"sub-01/**/*.tsv": {"sub-01/ses-meg/file.tsv", "sub-01/a/b/c/file.tsv"}},
        ),
        # Bare pattern without / expands as directory prefix
        (
            ["sub-01/file.tsv", "sub-01/ses-meg/file.tsv", "sub-010/file.tsv"],
            ["sub-01"],
            {"sub-01": {"sub-01/file.tsv", "sub-01/ses-meg/file.tsv"}},
        ),
        # Bare wildcard pattern expands as directory prefix
        (
            [
                "sub-01/file.tsv",
                "sub-02/file.tsv",
                "sub-010/file.tsv",
                "participants.tsv",
            ],
            ["sub-0?"],
            {
                "sub-0?": {
                    "sub-01/file.tsv",
                    "sub-02/file.tsv",
                },
            },
        ),
        # ** at end
        (
            ["sub-01/anything/here", "sub-02/other"],
            ["sub-01/**"],
            {"sub-01/**": {"sub-01/anything/here"}},
        ),
        # **/*.tsv matches .tsv files at any depth
        (
            ["participants.tsv", "sub-01/file.tsv", "sub-01/ses-meg/file.tsv"],
            ["**/*.tsv"],
            {
                "**/*.tsv": {
                    "participants.tsv",
                    "sub-01/file.tsv",
                    "sub-01/ses-meg/file.tsv",
                }
            },
        ),
        # * alone matches everything via directory expansion
        (
            ["participants.tsv", "sub-01/file.tsv"],
            ["*"],
            {"*": {"participants.tsv", "sub-01/file.tsv"}},
        ),
        # Combined include/exclude scenario
        (
            ["sub-01/a.tsv", "sub-01/b.nii", "sub-02/a.tsv"],
            ["sub-01/**/*.tsv"],
            {"sub-01/**/*.tsv": {"sub-01/a.tsv"}},
        ),
        # No match returns empty set
        (
            ["sub-01/file.tsv"],
            ["sub-99"],
            {"sub-99": set()},
        ),
        # MATCHBASE: bare *.ext matches at any depth (gitignore semantics)
        (
            [
                "sub-01/meg/run.fif",
                "sub-01/ses-meg/meg/run.fif",
                "root.fif",
            ],
            ["*.fif"],
            {
                "*.fif": {
                    "sub-01/meg/run.fif",
                    "sub-01/ses-meg/meg/run.fif",
                    "root.fif",
                }
            },
        ),
        # *.tsv matches at any depth via MATCHBASE
        (
            ["participants.tsv", "sub-01/file.tsv", "sub-01/ses-meg/file.tsv"],
            ["*.tsv"],
            {
                "*.tsv": {
                    "participants.tsv",
                    "sub-01/file.tsv",
                    "sub-01/ses-meg/file.tsv",
                }
            },
        ),
        # Directory path with / expands via /**
        (
            [
                "sub-0001/anat/T1w.nii",
                "sub-0001/anat/bold.json",
                "sub-0001/func/run.nii",
            ],
            ["sub-0001/anat"],
            {
                "sub-0001/anat": {
                    "sub-0001/anat/T1w.nii",
                    "sub-0001/anat/bold.json",
                }
            },
        ),
        # Trailing slash pattern
        (
            ["sub-01/file.tsv", "sub-01/ses-meg/file.tsv"],
            ["sub-01/"],
            {"sub-01/": {"sub-01/file.tsv", "sub-01/ses-meg/file.tsv"}},
        ),
        # Anchored pattern with / disables MATCHBASE
        (
            ["participants.tsv", "sub-01/file.tsv"],
            ["/*.tsv"],
            {"/*.tsv": {"participants.tsv"}},
        ),
    ],
)
def test_glob_filter(
    filenames: list[str],
    patterns: list[str],
    expected: dict[str, set[str]],
):
    """Test _glob.glob_filter against various patterns."""
    from openneuro._glob import glob_filter

    result = glob_filter(filenames, patterns)
    assert result == expected


# -- _safe_query tests --


@pytest.fixture
def _no_token():
    """Stub out get_token so _safe_query skips authentication."""
    with patch("openneuro._download.get_token", side_effect=ValueError):
        yield


@pytest.fixture
def _mock_gql_response(request):
    """Patch niquests.Session to return a mock response from _safe_query.

    Use `@pytest.mark.parametrize("_mock_gql_response", [...], indirect=True)`
    to set `status_code` and, optionally, `json_data` or `json_error`.
    """
    params = request.param
    mock_response = MagicMock()
    mock_response.status_code = params["status_code"]
    if "json_error" in params:
        mock_response.json.side_effect = params["json_error"]
    else:
        mock_response.json.return_value = params.get("json_data")

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("openneuro._download.niquests.Session", return_value=mock_client):
        yield mock_client


@pytest.mark.parametrize(
    "_mock_gql_response",
    [{"status_code": 200, "json_data": {"data": {"dataset": {}}}}],
    indirect=True,
)
@pytest.mark.usefixtures("_no_token")
def test_safe_query_posts_json_payload(_mock_gql_response):
    """Test that _safe_query sends a correct JSON POST to the GraphQL endpoint."""
    result, timed_out = _download._safe_query("query { test }")

    assert result == {"data": {"dataset": {}}}
    assert timed_out is False
    _mock_gql_response.post.assert_called_once_with(
        _download.gql_url,
        json={"query": "query { test }"},
        timeout=None,
        headers=_download.user_agent_header,
        cookies={},
    )


@pytest.mark.parametrize(
    "_mock_gql_response",
    [{"status_code": 502}],
    indirect=True,
)
@pytest.mark.usefixtures("_no_token", "_mock_gql_response")
def test_safe_query_retries_on_retryable_status():
    """Test that _safe_query returns (None, True) for retryable HTTP status codes."""
    result, timed_out = _download._safe_query("query { test }")

    assert result is None
    assert timed_out is True


@pytest.mark.parametrize(
    "_mock_gql_response",
    [{"status_code": 401, "json_error": json.JSONDecodeError("", "", 0)}],
    indirect=True,
)
@pytest.mark.usefixtures("_no_token", "_mock_gql_response")
def test_safe_query_raises_on_non_retryable_non_json():
    """_safe_query raises RuntimeError for non-retryable non-JSON responses."""
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _download._safe_query("query { test }")


def _make_fake_client(
    *,
    file_content: bytes,
    fail_head_n_times: int = 0,
    fail_head_status_code: int | None = None,
):
    """Create a mock `niquests.AsyncSession` for download tests.

    Parameters
    ----------
    file_content
        Bytes the fake GET response will yield.
    fail_head_n_times
        Number of initial HEAD requests that fail before succeeding.
    fail_head_status_code
        If set, failing HEAD requests return this HTTP status code instead
        of raising `niquests.ReadTimeout`.

    """
    head_call_count = 0

    async def head(url, *, headers=None, timeout=None):
        nonlocal head_call_count
        head_call_count += 1
        if head_call_count <= fail_head_n_times:
            if fail_head_status_code is not None:
                resp = MagicMock()
                resp.status_code = fail_head_status_code
                resp.ok = fail_head_status_code < 400
                resp.headers = {}
                return resp
            raise niquests.ReadTimeout("simulated timeout")
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.headers = {
            "etag": '"d41d8cd98f00b204e9800998ecf8427e"',
        }
        return resp

    class _FakeStream:
        def __init__(self):
            self.ok = True
            self.status_code = 200

        async def iter_content(self, chunk_size=-1, decode_unicode=False):
            async def gen():
                yield file_content

            return gen()

        async def close(self):
            pass

    async def get(url, *, headers=None, timeout=None, stream=None):
        return _FakeStream()

    client = AsyncMock()
    client.head = head
    client.get = get
    return client


def test_max_concurrent_downloads_validation(tmp_path: Path):
    """max_concurrent_downloads must be at least 1."""
    with pytest.raises(ValueError, match="max_concurrent_downloads must be at least 1"):
        download(dataset="ds000117", target_dir=tmp_path, max_concurrent_downloads=0)


def test_max_concurrent_downloads_cli_validation():
    """The CLI should reject --max-concurrent-downloads < 1."""
    from typer.testing import CliRunner

    from openneuro._cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["download", "--dataset=ds000117", "--max-concurrent-downloads=0"],
    )
    assert result.exit_code == 2


def test_multiple_include_options_cli():
    """The CLI should honor multiple --include (and --exclude) options."""
    from typer.testing import CliRunner

    from openneuro._cli import app

    runner = CliRunner()
    with patch("openneuro._cli.download") as mock_download:
        result = runner.invoke(
            app,
            [
                "download",
                "--dataset=ds000117",
                "--include=sub-0001",
                "--include=sub-0002",
                "--exclude=*.fif",
                "--exclude=*.json",
            ],
        )

    assert result.exit_code == 0
    mock_download.assert_called_once()
    _, kwargs = mock_download.call_args
    assert kwargs["include"] == ["sub-0001", "sub-0002"]
    assert kwargs["exclude"] == ["*.fif", "*.json"]


@contextmanager
def _progress_and_task() -> Iterator[tuple[Progress, TaskID]]:
    """Yield a disabled ``rich`` progress and an "overall" task (no output)."""
    with Progress(disable=True) as progress:
        yield progress, progress.add_task("overall", total=None)


def _run_download_file(
    tmp_path: Path,
    mock_client: AsyncMock,
    *,
    semaphore: asyncio.Semaphore | None = None,
    head_semaphore: asyncio.Semaphore | None = None,
    remote_file_size: int | None = 5,
) -> _download._DownloadStats:
    """Run `_download_file` with a mocked client; return the download stats."""
    if semaphore is None:
        semaphore = asyncio.Semaphore(2)
    if head_semaphore is None:
        head_semaphore = asyncio.Semaphore(_download._MAX_CONCURRENT_HEAD_REQUESTS)
    stats = _download._DownloadStats()

    async def run():
        with _progress_and_task() as (progress, overall_task):
            await _download_file(
                client=mock_client,
                url="https://example.com/test.txt",
                remote_file_size=remote_file_size,
                outfile=tmp_path / "test.txt",
                remote_path="test.txt",
                verify_hash=False,
                verify_size=False,
                max_retries=3,
                retry_backoff=0.0,
                semaphore=semaphore,
                head_semaphore=head_semaphore,
                query_str="test query",
                progress=progress,
                overall_task=overall_task,
                stats=stats,
            )

    asyncio.run(run())
    return stats


def test_semaphore_not_leaked_on_retry(tmp_path: Path):
    """Semaphore value must be preserved after retries.

    Regression test: the old recursive _retry_download() would call
    semaphore.release() explicitly, then the enclosing `async with
    semaphore:` would release again on exit — inflating the counter
    on every retry.
    """
    semaphore = asyncio.Semaphore(2)
    head_semaphore = asyncio.Semaphore(_download._MAX_CONCURRENT_HEAD_REQUESTS)
    mock_client = _make_fake_client(file_content=b"hello", fail_head_n_times=1)

    _run_download_file(
        tmp_path,
        mock_client,
        semaphore=semaphore,
        head_semaphore=head_semaphore,
    )

    assert semaphore._value == 2, (
        f"Semaphore leaked: expected value 2, got {semaphore._value}"
    )
    assert head_semaphore._value == _download._MAX_CONCURRENT_HEAD_REQUESTS, (
        f"HEAD semaphore leaked: expected value "
        f"{_download._MAX_CONCURRENT_HEAD_REQUESTS}, "
        f"got {head_semaphore._value}"
    )


def test_head_retryable_status_code(tmp_path: Path):
    """A retryable HEAD status code (e.g. 503) should be retried."""
    mock_client = _make_fake_client(
        file_content=b"hello",
        fail_head_n_times=1,
        fail_head_status_code=503,
    )

    _run_download_file(tmp_path, mock_client)

    assert (tmp_path / "test.txt").read_bytes() == b"hello"


def test_head_non_retryable_status_code(tmp_path: Path):
    """A non-retryable HEAD status code (e.g. 404) should raise _DownloadError."""
    mock_client = _make_fake_client(
        file_content=b"hello",
        fail_head_n_times=99,
        fail_head_status_code=404,
    )

    with pytest.raises(
        _download._DownloadError, match="HEAD request failed with HTTP 404"
    ):
        _run_download_file(tmp_path, mock_client)


# -- _retrieve_and_write_to_disk with remote_file_size=None --


def _mock_response(content: bytes) -> AsyncMock:
    response = AsyncMock()

    async def iter_content(chunk_size=-1, decode_unicode=False):
        async def gen():
            yield content

        return gen()

    response.iter_content = iter_content
    return response


def test_retrieve_and_write_to_disk_none_size(tmp_path: Path):
    """verify_size=True with remote_file_size=None must not crash."""
    outfile = tmp_path / "test.txt"
    content = b"hello world"

    with _progress_and_task() as (progress, overall_task):
        asyncio.run(
            _retrieve_and_write_to_disk(
                response=_mock_response(content),
                outfile=outfile,
                remote_path="test.txt",
                mode="wb",
                desc="test",
                local_file_size=0,
                remote_file_size=None,
                remote_file_hash=None,
                verify_hash=False,
                verify_size=True,
                progress=progress,
                overall_task=overall_task,
            )
        )
    assert outfile.read_bytes() == content


# -- download summary stats (gh-322) --


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 kB"),
        (1536, "1.5 kB"),
        (1048576, "1.0 MB"),
    ],
)
def test_format_size(num_bytes: int, expected: str):
    """`_format_size` renders human-readable, space-separated sizes."""
    assert _format_size(num_bytes) == expected


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        # `encoding` is None here, and `.lower()` on it used to raise at import.
        (io.StringIO(), False),
        (MagicMock(encoding="UTF-8"), True),
        (MagicMock(encoding="ascii"), False),
    ],
)
def test_probe_unicode(stream: object, expected: bool):
    """The probe tolerates streams that report no usable encoding."""
    with patch.object(sys, "stderr", stream):
        assert _download._probe_unicode() is expected


def test_download_file_updates_stats(tmp_path: Path):
    """Downloaded files are tallied; already-present files are not (gh-322)."""
    content = b"hello"
    client = _make_fake_client(file_content=content)

    # First run downloads the file: one file, len(content) bytes.
    stats = _run_download_file(tmp_path, client, remote_file_size=len(content))
    assert (stats.n_files, stats.n_bytes) == (1, len(content))

    # Second run finds a matching local file and downloads nothing.
    stats = _run_download_file(tmp_path, client, remote_file_size=len(content))
    assert (stats.n_files, stats.n_bytes) == (0, 0)


# -- shared-client connection bounding (gh-317) --


async def _serve_counting_http(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    conn_stats: dict[str, int],
    body: bytes,
) -> None:
    """Minimal keep-alive HTTP/1.1 handler that tracks open connections."""
    conn_stats["open"] += 1
    conn_stats["peak"] = max(conn_stats["peak"], conn_stats["open"])
    try:
        while True:
            request = await reader.readuntil(b"\r\n\r\n")
            method = request.split(b" ", 1)[0].decode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body))
            if method == "GET":
                writer.write(body)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass  # Client closed the connection.
    finally:
        conn_stats["open"] -= 1
        writer.close()


def test_connections_bounded_by_pool_not_file_count(tmp_path: Path):
    """Open connections must stay bounded by the pool size, not the file count.

    Regression test for gh-317: a per-file `niquests.AsyncSession` meant every
    dispatched task held an open connection while waiting for a download
    slot, so connection count grew with the number of files in the dataset.
    """
    max_concurrent_downloads = 3
    connection_bound = (
        max_concurrent_downloads + _download._MAX_CONCURRENT_HEAD_REQUESTS
    )
    n_files = 3 * connection_bound  # Enough that per-file clients would exceed it.
    body = b"0123456789"
    conn_stats = {"open": 0, "peak": 0}

    async def run() -> None:
        server = await asyncio.start_server(
            lambda r, w: _serve_counting_http(r, w, conn_stats, body),
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]
        files = [
            DatasetFile(
                filename=f"file_{i}.txt",
                urls=[f"http://127.0.0.1:{port}/file_{i}.txt"],
                size=len(body),
                id=f"id_{i}",
            )
            for i in range(n_files)
        ]
        async with server:
            await _download._download_files(
                target_dir=tmp_path,
                files=files,
                verify_hash=False,
                verify_size=True,
                max_retries=0,
                retry_backoff=0.0,
                max_concurrent_downloads=max_concurrent_downloads,
                query_str="test",
                stats=_download._DownloadStats(),
            )

    asyncio.run(run())

    assert conn_stats["peak"] <= connection_bound, (
        f"Peak connections ({conn_stats['peak']}) exceeded the pool bound "
        f"({connection_bound}) for {n_files} files"
    )
    for i in range(n_files):
        assert (tmp_path / f"file_{i}.txt").read_bytes() == body


def test_size_mismatch_uses_remote_path(tmp_path: Path):
    """Error message must contain remote_path, not the local outfile path."""
    remote_path = "sub-01/meg/file.fif"
    with pytest.raises(_download._RetryableError, match=remote_path) as exc_info:
        with _progress_and_task() as (progress, overall_task):
            asyncio.run(
                _retrieve_and_write_to_disk(
                    response=_mock_response(b"hello"),
                    outfile=tmp_path / "test.txt",
                    remote_path=remote_path,
                    mode="wb",
                    desc="test",
                    local_file_size=0,
                    remote_file_size=999_999,  # intentional mismatch
                    remote_file_hash=None,
                    verify_hash=False,
                    verify_size=True,
                    progress=progress,
                    overall_task=overall_task,
                )
            )
    assert str(tmp_path) not in str(exc_info.value)


# -- blocking coroutine execution, incl. when a loop is already running (gh-329) --


@pytest.mark.parametrize("loop_already_running", [False, True])
def test_run_coroutine_blocking(loop_already_running: bool):
    """Block until the coroutine finishes and surface its errors (gh-329).

    Covers both the plain `asyncio.run` path and the worker-thread path used
    when a loop is already running (e.g. in Jupyter), where the old
    fire-and-forget `loop.create_task` returned early -- leaving the download
    stats empty and swallowing failures.
    """
    ran: list[bool] = []

    async def _ok() -> None:
        await asyncio.sleep(0)
        ran.append(True)

    async def _boom() -> None:
        raise ValueError("boom")

    def _run(coro) -> None:
        if not loop_already_running:
            _download._run_coroutine_blocking(coro)
            return

        async def _driver() -> None:
            _download._run_coroutine_blocking(coro)

        asyncio.run(_driver())

    _run(_ok())
    assert ran == [True]  # returned only after the coroutine completed

    with pytest.raises(ValueError, match="boom"):
        _run(_boom())
