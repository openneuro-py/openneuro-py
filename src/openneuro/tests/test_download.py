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
from openneuro.tests.utils import load_json

dataset_id_aws = "ds000246"
tag_aws = "1.0.0"
include_aws = "sub-0001/anat"
exclude_aws = []

dataset_id_on = "ds000117"
tag_on = None
include_on = "sub-16/ses-meg"
exclude_on = "*.fif"  # save GBs of downloads

invalid_tag = "abcdefg"


@pytest.mark.parametrize(
    ("dataset_id", "tag", "include", "exclude"),
    [
        (dataset_id_aws, tag_aws, include_aws, exclude_aws),
        (dataset_id_on, tag_on, include_on, exclude_on),
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
    tmp_path: Path, dataset_id=dataset_id_aws, invalid_tag=invalid_tag
):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match="snapshot.*does not exist"):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)


def test_resume_download(tmp_path: Path):
    """Test resuming of a dataset download."""
    dataset = "ds000246"
    tag = "1.0.0"
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


@pytest.mark.parametrize(
    ("dir_path", "include_pattern", "expected"),
    load_json("traverse_test_cases.json"),
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
    dataset: str, include: list[str], expected_files: list[str]
):
    """Test that download generates the correct list of files.

    This test verifies the file filtering logic by mocking the
    metadata retrieval and checking that the correct files are
    selected based on include/exclude patterns.
    """
    MOCK_METADATA = load_json("mock_metadata_ds000117.json")

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
        # Run the function with an include pattern
        _download.download(
            dataset=dataset,
            target_dir=Path("/tmp/test"),
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
def test_download_file_count(dataset: str, include: list[str], expected_num_files: int):
    """Test that download generates the correct number of files.

    This test verifies the file filtering logic by mocking
    the metadata retrieval and checking that the correct
    number of files are selected based on include patterns.
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
            target_dir=Path("/tmp/test"),
            include=include,
        )

        files_arg = _download_files_spy.call_args[1]["files"]
        files_arg = [file["filename"] for file in files_arg]
        assert len(files_arg) == expected_num_files, (
            f"Expected {expected_num_files} files, got {len(files_arg)}"
        )
