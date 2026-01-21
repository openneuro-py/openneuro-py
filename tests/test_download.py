"""Test downloading and authentication."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import openneuro
import openneuro._config
from openneuro import _download
from openneuro._download import (
    _traverse_directory,
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

    # We should be able to resume a download even if "datset_description.jon"
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


_TEST_CASES_LIST = load_json("traverse_dir_test_cases.json")
assert isinstance(_TEST_CASES_LIST, list)


@pytest.mark.parametrize(
    ("dir_path", "include_pattern", "expected"),
    _TEST_CASES_LIST
    + [
        # TODO: These three tests cases are failing because directory
        # should not be traversed for include_pattern that does not
        # match dir_path itself
        pytest.param(
            "sub-01/ses-meg",
            "sub-01/ses-meg/*.tsv",
            False,
            marks=pytest.mark.xfail(
                reason="Known bug: directory should not be traversed for file"
                "pattern that does not match directory itself"
            ),
        ),
        pytest.param(
            "sub-01/ses-meg/meg",
            "sub-01/ses-meg/*.tsv",
            False,
            marks=pytest.mark.xfail(
                reason="Known bug: directory should not be traversed for file"
                "pattern that does not match directory itself"
            ),
        ),
        pytest.param(
            "sub-01/ses-meg/meg",
            "*/*.json",
            False,
            marks=pytest.mark.xfail(
                reason="Known bug: directory should not be traversed for file"
                "pattern that does not match directory itself"
            ),
        ),
    ],
)
def test_traverse_directory(
    dir_path: str,
    include_pattern: str,
    expected: bool,
):
    """Test that the right directories are traversed.

    This test uses realistic OpenNeuro directory structures
    following BIDS standards, and tests against a comprehensive
    set of include patterns commonly used in practice. It checks
    if the right directories are traversed based on the include
    pattern.

    Test cases are loaded from `traverse_test_cases.json` which
    contains an array of test case tuples. Each tuple has the
    structure:
    [dir_path, include_pattern, expected_result]

    Where:
    - dir_path: Directory path from OpenNeuro dataset (e.g.,
      "sub-01", "sub-01/ses-meg", "derivatives")
    - include_pattern: Glob pattern to match (e.g., "*.tsv",
      "sub-01/**", "**/meg/**")
    - expected_result: Boolean indicating if directory should
      be traversed (true/false)

    To add more test cases:
    1. Open `src/openneuro/tests/data/traverse_test_cases.json`
    2. Add new test case as: ["dir_path", "pattern", true/false]
    3. Ensure JSON syntax is valid (commas, quotes, brackets)
    4. Test cases should cover edge cases and common patterns

    Parameters
    ----------
    dir_path : str
        The directory path from a realistic OpenNeuro dataset.
    include_pattern : str
        The include pattern to match against
    expected : bool
        Expected result (True if directory should be traversed)

    """
    result = _traverse_directory(dir_path, include_pattern)
    assert result == expected, (
        f"_traverse_directory(dir_path={dir_path}, include_pattern={include_pattern}) "
        f"returned {result}, expected {expected}"
    )


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
    filenames, URLs, sizes, and directory flags for realistic
    testing without requiring actual API calls. Having a mock
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
        tree = kwargs.get("tree", "null").strip('"').strip("'")
        return copy.deepcopy(MOCK_METADATA[tree])

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
        # Load mock metadata
        MOCK_METADATA = load_json(f"mock_metadata_{dataset}.json")

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

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments."""
        return None

    with patch.object(
        _download, "_download_files", side_effect=_download_files_spy
    ) as _download_files_spy:
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
