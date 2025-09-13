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

TRAVERSE_TEST_CASES = [
    ('derivatives', '*', True),
    ('derivatives', 'dataset_description.json', False),
    ('derivatives', '*.json', False),
    ('derivatives', 'sub-01', False),
    ('derivatives', 'sub-01/', False),
    ('derivatives', 'sub-02', False),
    ('derivatives', 'sub-02/', False),
    ('derivatives', 'sub-01/*', False),
    ('derivatives', 'sub-01/**', False),
    ('derivatives', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives', 'sub-01/**/*.tsv', False),
    ('derivatives', 'sub-*/**/*.tsv', False),
    ('derivatives', 'sub-*', False),
    ('derivatives', 'sub-*/', False),
    ('derivatives', 'sub-*/**/meg/**', False),
    ('derivatives', '**/meg/**', True),
    ('derivatives', '**/*.json', True),
    ('derivatives/meg_derivatives', '*', True),
    ('derivatives/meg_derivatives', 'dataset_description.json', False),
    ('derivatives/meg_derivatives', '*.json', False),
    ('derivatives/meg_derivatives', 'sub-01', False),
    ('derivatives/meg_derivatives', 'sub-01/', False),
    ('derivatives/meg_derivatives', 'sub-02', False),
    ('derivatives/meg_derivatives', 'sub-02/', False),
    ('derivatives/meg_derivatives', 'sub-01/*', False),
    ('derivatives/meg_derivatives', 'sub-01/**', False),
    ('derivatives/meg_derivatives', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives', 'sub-*', False),
    ('derivatives/meg_derivatives', 'sub-*/', False),
    ('derivatives/meg_derivatives', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives', '**/meg/**', True),
    ('derivatives/meg_derivatives', '**/*.json', True),
    ('derivatives/meg_derivatives/ct_sparse.fif', '*', True),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', '*.json', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-02', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-02/', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/*', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/**', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-*', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-*/', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/ct_sparse.fif', '**/meg/**', True),
    ('derivatives/meg_derivatives/ct_sparse.fif', '**/*.json', True),
    ('derivatives/meg_derivatives/sss_cal.dat', '*', True),
    ('derivatives/meg_derivatives/sss_cal.dat', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sss_cal.dat', '*.json', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-02', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-02/', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-*', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-*/', False),
    ('derivatives/meg_derivatives/sss_cal.dat', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sss_cal.dat', '**/meg/**', True),
    ('derivatives/meg_derivatives/sss_cal.dat', '**/*.json', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg', '*', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', '*.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-02', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-02/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-*/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg', '**/meg/**', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg', '**/*.json', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '*', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '*.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-02', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-02/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '**/meg/**', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '**/*.json', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', '*', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', '*.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-02', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-02/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-*/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', '**/meg/**', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg', '**/*.json', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '*', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '*.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-02', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-02/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '**/meg/**', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_proc-sss_meg.json', '**/*.json', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', '*', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'dataset_description.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', '*.json', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-02', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-02/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/ses-meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/ses-meg/meg/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-01/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-*/**/*.tsv', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-*', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-*/', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', 'sub-*/**/meg/**', False),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', '**/meg/**', True),
    ('derivatives/meg_derivatives/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif', '**/*.json', True),
    ('sub-01', '*', True),
    ('sub-01', 'dataset_description.json', False),
    ('sub-01', '*.json', False),
    ('sub-01', 'sub-01', True),
    ('sub-01', 'sub-01/', True),
    ('sub-01', 'sub-02', False),
    ('sub-01', 'sub-02/', False),
    ('sub-01', 'sub-01/*', True),
    ('sub-01', 'sub-01/**', True),
    ('sub-01', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01', 'sub-01/**/*.tsv', True),
    ('sub-01', 'sub-*/**/*.tsv', True),
    ('sub-01', 'sub-*', True),
    ('sub-01', 'sub-*/', True),
    ('sub-01', 'sub-*/**/meg/**', True),
    ('sub-01', '**/meg/**', True),
    ('sub-01', '**/*.json', True),
    ('sub-01/ses-meg', '*', True),
    ('sub-01/ses-meg', 'dataset_description.json', False),
    ('sub-01/ses-meg', '*.json', False),
    ('sub-01/ses-meg', 'sub-01', True),
    ('sub-01/ses-meg', 'sub-01/', True),
    ('sub-01/ses-meg', 'sub-02', False),
    ('sub-01/ses-meg', 'sub-02/', False),
    ('sub-01/ses-meg', 'sub-01/*', True),
    ('sub-01/ses-meg', 'sub-01/**', True),
    ('sub-01/ses-meg', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg', 'sub-*', True),
    ('sub-01/ses-meg', 'sub-*/', True),
    ('sub-01/ses-meg', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg', '**/meg/**', True),
    ('sub-01/ses-meg', '**/*.json', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', '*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'dataset_description.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', '*.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-02', False),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-02/', False),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-*/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', '**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_scans.tsv', '**/*.json', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', '*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'dataset_description.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', '*.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-02', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-02/', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-*/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', '**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv', '**/*.json', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', '*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'dataset_description.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', '*.json', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-02', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-02/', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-*', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-*/', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', '**/meg/**', True),
    ('sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json', '**/*.json', True),
    ('sub-01/ses-meg/beh', '*', True),
    ('sub-01/ses-meg/beh', 'dataset_description.json', False),
    ('sub-01/ses-meg/beh', '*.json', False),
    ('sub-01/ses-meg/beh', 'sub-01', True),
    ('sub-01/ses-meg/beh', 'sub-01/', True),
    ('sub-01/ses-meg/beh', 'sub-02', False),
    ('sub-01/ses-meg/beh', 'sub-02/', False),
    ('sub-01/ses-meg/beh', 'sub-01/*', True),
    ('sub-01/ses-meg/beh', 'sub-01/**', True),
    ('sub-01/ses-meg/beh', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/beh', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-meg/beh', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/beh', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/beh', 'sub-*', True),
    ('sub-01/ses-meg/beh', 'sub-*/', True),
    ('sub-01/ses-meg/beh', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/beh', '**/meg/**', True),
    ('sub-01/ses-meg/beh', '**/*.json', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', '*', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'dataset_description.json', False),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', '*.json', False),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-02', False),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-02/', False),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/*', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/**', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-*', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-*/', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', '**/meg/**', True),
    ('sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv', '**/*.json', True),
    ('sub-01/ses-meg/meg', '*', True),
    ('sub-01/ses-meg/meg', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg', '*.json', False),
    ('sub-01/ses-meg/meg', 'sub-01', True),
    ('sub-01/ses-meg/meg', 'sub-01/', True),
    ('sub-01/ses-meg/meg', 'sub-02', False),
    ('sub-01/ses-meg/meg', 'sub-02/', False),
    ('sub-01/ses-meg/meg', 'sub-01/*', True),
    ('sub-01/ses-meg/meg', 'sub-01/**', True),
    ('sub-01/ses-meg/meg', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg', 'sub-*', True),
    ('sub-01/ses-meg/meg', 'sub-*/', True),
    ('sub-01/ses-meg/meg', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg', '**/meg/**', True),
    ('sub-01/ses-meg/meg', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv', '**/*.json', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', '*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'dataset_description.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', '*.json', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-02', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-02/', False),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/ses-meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/ses-meg/meg/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-*', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-*/', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', 'sub-*/**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', '**/meg/**', True),
    ('sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif', '**/*.json', True),
    ('sub-01/ses-mri', '*', True),
    ('sub-01/ses-mri', 'dataset_description.json', False),
    ('sub-01/ses-mri', '*.json', False),   
    ('sub-01/ses-mri', 'sub-01', True),
    ('sub-01/ses-mri', 'sub-01/', True),
    ('sub-01/ses-mri', 'sub-02', False),
    ('sub-01/ses-mri', 'sub-02/', False),
    ('sub-01/ses-mri', 'sub-01/*', True),
    ('sub-01/ses-mri', 'sub-01/**', True),
    ('sub-01/ses-mri', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri', 'sub-*', True),
    ('sub-01/ses-mri', 'sub-*/', True),
    ('sub-01/ses-mri', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri', '**/meg/**', True),
    ('sub-01/ses-mri', '**/*.json', True),
    ('sub-01/ses-mri/anat', '*', True),
    ('sub-01/ses-mri/anat', 'dataset_description.json', False),
    ('sub-01/ses-mri/anat', '*.json', False),
    ('sub-01/ses-mri/anat', 'sub-01', True),
    ('sub-01/ses-mri/anat', 'sub-01/', True),
    ('sub-01/ses-mri/anat', 'sub-02', False),
    ('sub-01/ses-mri/anat', 'sub-02/', False),
    ('sub-01/ses-mri/anat', 'sub-01/*', True),
    ('sub-01/ses-mri/anat', 'sub-01/**', True),
    ('sub-01/ses-mri/anat', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/anat', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/anat', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/anat', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/anat', 'sub-*', True),
    ('sub-01/ses-mri/anat', 'sub-*/', True),
    ('sub-01/ses-mri/anat', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/anat', '**/meg/**', True),
    ('sub-01/ses-mri/anat', '**/*.json', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', '*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'dataset_description.json', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', '*.json', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-02', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-02/', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-*/', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', '**/meg/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz', '**/*.json', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', '*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'dataset_description.json', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', '*.json', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-02', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-02/', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-*', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-*/', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', '**/meg/**', True),
    ('sub-01/ses-mri/anat/sub-01_ses-mri_T1w.json', '**/*.json', True),
    ('sub-01/ses-mri/func', '*', True),
    ('sub-01/ses-mri/func', 'dataset_description.json', False),
    ('sub-01/ses-mri/func', '*.json', False),
    ('sub-01/ses-mri/func', 'sub-01', True),
    ('sub-01/ses-mri/func', 'sub-01/', True),
    ('sub-01/ses-mri/func', 'sub-02', False),
    ('sub-01/ses-mri/func', 'sub-02/', False),
    ('sub-01/ses-mri/func', 'sub-01/*', True),
    ('sub-01/ses-mri/func', 'sub-01/**', True),
    ('sub-01/ses-mri/func', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/func', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/func', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/func', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/func', 'sub-*', True),
    ('sub-01/ses-mri/func', 'sub-*/', True),
    ('sub-01/ses-mri/func', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/func', '**/meg/**', True),
    ('sub-01/ses-mri/func', '**/*.json', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', '*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'dataset_description.json', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', '*.json', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-02', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-02/', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-*/', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', '**/meg/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.nii.gz', '**/*.json', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', '*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'dataset_description.json', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', '*.json', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-02', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-02/', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/ses-meg/*.tsv', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-01/**/*.tsv', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-*/**/*.tsv', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-*', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-*/', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', 'sub-*/**/meg/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', '**/meg/**', True),
    ('sub-01/ses-mri/func/sub-01_ses-mri_task-faces_bold.json', '**/*.json', True),
    ('sub-02', '*', True),
    ('sub-02', 'dataset_description.json', False),
    ('sub-02', '*.json', False),
    ('sub-02', 'sub-01', False),
    ('sub-02', 'sub-01/', False),
    ('sub-02', 'sub-02', True),
    ('sub-02', 'sub-02/', True),
    ('sub-02', 'sub-01/*', False),
    ('sub-02', 'sub-01/**', False),
    ('sub-02', 'sub-01/ses-meg/*.tsv', False),
    ('sub-02', 'sub-01/ses-meg/meg/*.tsv', False),
    ('sub-02', 'sub-01/**/*.tsv', False),
    ('sub-02', 'sub-*/**/*.tsv', True),
    ('sub-02', 'sub-*', True),
    ('sub-02', 'sub-*/', True),
    ('sub-02', 'sub-*/**/meg/**', True),
    ('sub-02', '**/meg/**', True),
    ('sub-02', '**/*.json', True),    
]
@pytest.mark.parametrize(
    ("dir_path", "include_pattern", "expected"),
    TRAVERSE_TEST_CASES,
)
def test_traverse_directory(
    dir_path: str, include_pattern: str, expected: bool,
):
    """Test _traverse_directory function with comprehensive OpenNeuro dataset patterns.
    
    This test uses realistic OpenNeuro directory structures following BIDS standards
    and tests against a comprehensive set of include patterns commonly used in practice.
    
    Parameters
    ----------
    dir_path : str
        The directory path from a realistic OpenNeuro dataset
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
