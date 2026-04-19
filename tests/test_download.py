"""Test downloading and authentication."""

import asyncio
import copy
import importlib
import json
import ssl
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import openneuro
import openneuro._config
from openneuro import _download
from openneuro._download import (
    _download_file,
    download,
)
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

    def mock_get_download_metadata(*args, **kwargs):
        return copy.deepcopy(MOCK_METADATA)

    def mock_get_local_tag(*args, **kwargs):
        return None

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments."""
        return None

    # Load mock metadata
    MOCK_METADATA = load_json(f"mock_metadata_{dataset}.json")

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
        files_arg = [file["filename"] for file in files_arg]
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

    def mock_get_download_metadata(*args, **kwargs):
        return copy.deepcopy(MOCK_METADATA)

    def mock_get_local_tag(*args, **kwargs):
        return None

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments."""
        return None

    # Load mock metadata
    MOCK_METADATA = load_json(f"mock_metadata_{dataset}.json")

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
        files_arg = [file["filename"] for file in files_arg]
        assert len(files_arg) == expected_num_files, (
            f"Expected {expected_num_files} files, got {len(files_arg)}"
        )


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


# -- SSL context tests --


@pytest.fixture
def _restore_ssl_context():
    """Restore SSL-related state after tests that reload _download.

    Restores ssl_context, _use_truststore, and truststore.
    """
    original_context = _download.ssl_context
    original_use_truststore = _download._use_truststore
    original_truststore = getattr(_download, "truststore", None)
    yield
    _download.ssl_context = original_context
    _download._use_truststore = original_use_truststore
    if original_truststore is not None:
        _download.truststore = original_truststore


def test_ssl_context_is_set():
    """Test that ssl_context is an ssl.SSLContext instance."""
    assert isinstance(_download.ssl_context, ssl.SSLContext)


@pytest.mark.usefixtures("_restore_ssl_context")
def test_ssl_fallback_on_import_error():
    """Test fallback to default SSL context when truststore is not installed."""
    with patch.dict(sys.modules, {"truststore": None}):
        with pytest.warns(match="Could not use truststore.*falling back"):
            mod = importlib.reload(_download)
        assert isinstance(mod.ssl_context, ssl.SSLContext)
        assert type(mod.ssl_context) is ssl.SSLContext


@pytest.mark.usefixtures("_restore_ssl_context")
def test_ssl_fallback_on_construction_error():
    """Test fallback when truststore imports but SSLContext() raises."""
    mock_truststore = MagicMock()
    mock_truststore.SSLContext.side_effect = OSError("backend failure")
    with patch.dict(sys.modules, {"truststore": mock_truststore}):
        with pytest.warns(match="Could not use truststore.*backend failure"):
            mod = importlib.reload(_download)
        assert isinstance(mod.ssl_context, ssl.SSLContext)
        assert type(mod.ssl_context) is ssl.SSLContext


# -- _safe_query tests --


@pytest.fixture
def _no_token():
    """Stub out get_token so _safe_query skips authentication."""
    with patch("openneuro._download.get_token", side_effect=ValueError):
        yield


@pytest.fixture
def _mock_gql_response(request):
    """Patch httpx.Client to return a mock response from _safe_query.

    Use ``@pytest.mark.parametrize("_mock_gql_response", [...], indirect=True)``
    to set ``status_code`` and, optionally, ``json_data`` or ``json_error``.
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

    with patch("openneuro._download.httpx.Client", return_value=mock_client):
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


def _make_fake_client(*, file_content: bytes, fail_head_n_times: int = 0):
    """Create a mock ``httpx.AsyncClient`` for download tests.

    Parameters
    ----------
    file_content
        Bytes the fake GET response will yield.
    fail_head_n_times
        Number of initial HEAD requests that raise ``httpx.ReadTimeout``
        before succeeding.

    """
    head_call_count = 0

    async def head(url, *, headers=None):
        nonlocal head_call_count
        head_call_count += 1
        if head_call_count <= fail_head_n_times:
            raise httpx.ReadTimeout("simulated timeout")
        resp = MagicMock()
        resp.headers = {
            "etag": '"d41d8cd98f00b204e9800998ecf8427e"',
            "content-length": str(len(file_content)),
        }
        return resp

    class _FakeStream:
        def __init__(self):
            self.is_error = False
            self.status_code = 200
            self.num_bytes_downloaded = len(file_content)

        async def aiter_bytes(self):
            yield file_content

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    client = AsyncMock()
    client.head = head
    client.stream = lambda method, *, url, headers=None: _FakeStream()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
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


def test_semaphore_not_leaked_on_retry(tmp_path: Path):
    """Semaphore value must be preserved after retries.

    Regression test: the old recursive _retry_download() would call
    semaphore.release() explicitly, then the enclosing ``async with
    semaphore:`` would release again on exit — inflating the counter
    on every retry.
    """
    semaphore = asyncio.Semaphore(2)
    mock_client = _make_fake_client(file_content=b"hello", fail_head_n_times=1)

    async def run():
        await _download_file(
            url="https://example.com/test.txt",
            api_file_size=5,
            outfile=tmp_path / "test.txt",
            verify_hash=False,
            verify_size=False,
            max_retries=3,
            retry_backoff=0.0,
            semaphore=semaphore,
            query_str="test query",
        )

    with patch("openneuro._download.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(run())

    assert semaphore._value == 2, (
        f"Semaphore leaked: expected value 2, got {semaphore._value}"
    )
