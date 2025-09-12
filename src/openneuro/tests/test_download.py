"""Test downloading and authentication."""

import json
from pathlib import Path
from unittest import mock

import pytest

import openneuro
import openneuro._config
from openneuro import download
from openneuro._download import _traverse_directory

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
    with mock.patch.object(openneuro._config, "CONFIG_PATH", tmp_path / ".openneuro"):
        with mock.patch("getpass.getpass", lambda _: openneuro_token):
            openneuro._config.init_config()

        # This is a restricted dataset that is only available if the API token
        # was used correctly.
        download(dataset="ds006412", include="README.txt", target_dir=tmp_path)

    assert (tmp_path / "README.txt").exists()


@pytest.mark.parametrize(
    ("dir_path", "include_pattern", "expected"),
    [
        # Test Case 1: Exact Directory Match
        ("sub-01", "sub-01", True),
        ("sub-02", "sub-01", False),
        ("sub-01/ses-meg", "sub-01/ses-meg", True),
        ("sub-01/ses-mri", "sub-01/ses-meg", False),
        
        # Test Case 2: Directory is a Parent of the Include Pattern
        ("sub-01", "sub-01/*", True),
        ("sub-01", "sub-01/ses-meg/*", True),
        ("sub-01/ses-mri", "sub-01/ses-meg/*", False),
        ("sub-01/ses-meg", "sub-01/ses-meg/*", True),
        ("sub-01/ses-meg/meg", "sub-01/ses-meg/*", True),
        ("sub-02", "sub-01/*", False),
        ("sub-01/ses-meg", "sub-01/ses-mri/*", False),
        
        # Test Case 3: Directory or Subdirectory Match (No Wildcards)
        ("sub-01", "sub-01/ses-emg", True),
        ("sub-01", "sub-01/ses-emg/", True),
        ("sub-01/ses-mri", "sub-01/ses-meg", False),
        ("sub-01/ses-emg", "sub-01/ses-emg", True),
        ("sub-01/ses-emg", "sub-01/ses-emg/", True),
        ("sub-01/ses-emg/meg", "sub-01/ses-emg", True),
        ("sub-01/ses-emg/meg", "sub-01/ses-emg/", True),
        ("sub-02/ses-emg", "sub-01/ses-emg", False),
        
        # Test Case 4: Wildcard Pattern Prefix Match
        ("sub-01/ses-meg", "sub-01/*", True),
        ("sub-01/ses-meg/meg", "sub-01/*", True),
        ("sub-02/ses-meg", "sub-01/*", False),
        ("sub-01/ses-meg", "sub-01/ses-*", True),
        ("sub-01/ses-mri", "sub-01/ses-*", True),
        ("sub-01/anat", "sub-01/ses-*", False),
        ("sub-01/ses-meg/meg", "sub-01/ses-*", True),
        ("sub-01/ses-meg/meg", "sub-01/ses-meg/*", True),
        ("sub-01/ses-meg/meg", "sub-01/ses-mri/*", False),
        
        # Edge Cases
        ("", "", True),  # Empty paths
        ("", "sub-01", True),
        # ("sub-01", "", True),
        ("sub-01", "sub-01/", True),  # Trailing slash
        ("sub-01/", "sub-01", True),  # Trailing slash on dir_path
        ("sub-01/", "sub-01/", True),  # Both with trailing slash
        
        # Deep nesting tests
        ("sub-01/ses-meg/meg/raw", "sub-01/*", True),
        ("sub-01/ses-meg/meg/raw", "sub-01/ses-*", True),
        ("sub-01/ses-meg/meg/raw", "sub-01/ses-meg/*", True),
        ("sub-01/ses-meg/meg/raw", "sub-01/ses-meg/meg/*", True),
        ("sub-01/ses-meg/meg/raw", "sub-01/ses-mri/*", False),
        ("sub-01/ses-meg/meg/raw", "sub-02/*", False),
        
        # Complex wildcard patterns
        ("sub-01/ses-meg", "sub-*", True),
        ("sub-01/ses-meg", "sub-01/ses-*", True),  # Wildcard in middle
        # ("sub-01/ses-meg", "sub-01/*meg*", False),  # Wildcard around text (not supported)
        # ("sub-01/ses-meg", "sub-01/ses-*meg", False),  # Wildcard before text (not supported)
        
        # Special characters and edge cases
        ("sub-01_special", "sub-01_special", True),
        ("sub-01-special", "sub-01-special", True),
        ("sub-01.special", "sub-01.special", True),
        ("sub-01_special", "sub-01-special", False),  # Different separators
        ("sub-01-special", "sub-01_special", False),  # Different separators
        
        # Multiple wildcards (should match prefix before first *)
        ("sub-01/ses-meg/meg", "sub-01/*/*", True),
        ("sub-01/ses-meg/meg", "sub-01/ses-*/*", True),
        # ("sub-01/ses-meg/meg", "sub-01/ses-*meg*", False),  # Multiple wildcards not supported
        
        # Very deep paths
        ("a/b/c/d/e/f/g/h/i/j", "a/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/g/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/g/h/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/g/h/i/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/g/h/i/j/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "a/b/c/d/e/f/g/h/i/j/k/*", True),
        ("a/b/c/d/e/f/g/h/i/j", "b/*", False),  # Wrong prefix
    ],
)
def test_traverse_directory(dir_path: str, include_pattern: str, expected: bool):
    """Test _traverse_directory function with various directory paths and include patterns.
    
    This comprehensive test covers all the different cases handled by the function:
    1. Exact directory match
    2. Directory is a parent of the include pattern
    3. Directory or subdirectory match (no wildcards)
    4. Wildcard pattern prefix match
    
    Parameters
    ----------
    dir_path : str
        The directory path to test
    include_pattern : str
        The include pattern to match against
    expected : bool
        Expected result (True if directory should be traversed)
    """
    result = _traverse_directory(dir_path, include_pattern)
    assert result == expected, (
        f"_traverse_directory('Directory: {dir_path}', 'Include Pattern: {include_pattern}') "
        f"returned {result}, expected {expected}"
    )
