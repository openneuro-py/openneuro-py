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


TRAVERSE_TEST_CASES = [
    ("sub-01", "*", True),
    ("sub-01", "*.json", False),
    ("sub-01", "dataset_description.json", False),
    ("sub-01", "sub-01", True),
    ("sub-01", "sub-01/", True),
    ("sub-01", "sub-01/*", True),
    ("sub-01", "sub-01/**", True),
    ("sub-01", "sub-01/ses-meg", True),
    ("sub-01", "sub-01/ses-meg/meg/*.tsv", True),
    ("sub-01", "sub-01/**/*.tsv", True),
    ("sub-01", "sub-*", True),
    ("sub-01", "sub-*/", True),
    ("sub-01", "sub-*/**", True),
    ("sub-01", "sub-*/**/*.tsv", True),
    ("sub-01", "sub-*/**/meg/**", True),
    ("sub-01", "**/meg/**", True),
    ("sub-01", "**/*.json", True),
    ("sub-01", "sub-02", False),
    ("sub-01", "sub-02/", False),
    ("sub-01", "sub-02/*", False),
    ("sub-01", "sub-02/**", False),
    ("sub-01/ses-meg", "sub-01", True),
    ("sub-01/ses-meg", "sub-01/ses-meg", True),
    ("sub-01/ses-meg", "sub-01/ses-meg/meg/*.tsv", True),
    ("sub-01/ses-meg", "sub-01/ses-meg/*.tsv", False),  # This is failing
    ("sub-01/ses-meg", "sub-01/ses-mri/", False),
    ("sub-01/ses-meg/meg", "sub-01/*/meg/*.tsv", True),
    ("sub-01/ses-meg/meg", "sub-01/**/meg/*.tsv", True),
    ("sub-01/ses-meg/meg", "sub-01/**/*.tsv", True),
    ("sub-01/ses-meg/meg", "sub-*/**/*.tsv", True),
    ("sub-01/ses-meg/meg", "sub-*/**/meg/**", True),
    ("sub-01/ses-meg/meg", "**/meg/**", True),
    ("sub-01/ses-meg/meg", "**/*.json", True),
    ("sub-01/ses-meg/meg", "*/*.json", False),  # This is failing
    ("sub-01/ses-meg/meg", "sub-01/ses-meg/*.tsv", False),  # This is failing
    ("derivatives", "sub-01", False),
    ("derivatives", "sub-01/", False),
    ("derivatives", "sub-01/*", False),
    ("derivatives", "sub-01/**", False),
    ("derivatives", "sub-01/**/*.tsv", False),
    ("derivatives", "sub-*", False),
    ("derivatives", "sub-*/**/meg/**", False),
    ("derivatives", "**/meg/**", True),
    ("derivatives", "**/*.json", True),
]


@pytest.mark.parametrize(
    ("dir_path", "include_pattern", "expected"),
    TRAVERSE_TEST_CASES,
)
def test_traverse_directory(
    dir_path: str,
    include_pattern: str,
    expected: bool,
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


MOCK_METADATA = {
    "null": {
        "id": "ds000117:1.1.0",
        "files": [
            # Root level files
            {
                "filename": "dataset_description.json",
                "urls": ["http://example.com/dataset_description.json"],
                "size": 1000,
                "directory": False,
                "id": "root1",
            },
            {
                "filename": "participants.tsv",
                "urls": ["http://example.com/participants.tsv"],
                "size": 500,
                "directory": False,
                "id": "root2",
            },
            {
                "filename": "participants.json",
                "urls": ["http://example.com/participants.json"],
                "size": 500,
                "directory": False,
                "id": "root3",
            },
            {
                "filename": "README",
                "urls": ["http://example.com/README"],
                "size": 2000,
                "directory": False,
                "id": "root4",
            },
            {
                "filename": "CHANGES",
                "urls": ["http://example.com/CHANGES"],
                "size": 300,
                "directory": False,
                "id": "root5",
            },
            # Subject directories
            {
                "filename": "derivatives",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives",
            },
            {
                "filename": "sub-01",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1",
            },
            {
                "filename": "sub-02",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2",
            },
            {
                "filename": "sub-emptyroom",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub-emptyroom",
            },
        ],
    },
    "derivatives": {
        "files": [
            {
                "filename": "freesurfer",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer",
            },
            {
                "filename": "meg_derivatives",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_meg_derivatives",
            },
        ]
    },
    "derivatives_meg_derivatives": {
        "files": [
            {
                "filename": "ct_sparse.fif",
                "urls": [
                    "http://example.com/derivatives/meg_derivatives/ct_sparse.fif"
                ],
                "size": 1000,
                "directory": False,
                "id": "derivatives_meg_derivatives_ct_sparse",
            },
            {
                "filename": "sss_cal.dat",
                "urls": ["http://example.com/derivatives/meg_derivatives/sss_cal.dat"],
                "size": 2000,
                "directory": False,
                "id": "derivatives_meg_derivatives_sss_cal",
            },
        ]
    },
    "derivatives_freesurfer": {
        "files": [
            {
                "filename": "README",
                "urls": ["http://example.com/derivatives/freesurfer/file1.txt"],
                "size": 123,
                "directory": False,
                "id": "derivatives_freesurfer_file1",
            },
            {
                "filename": "sub-01",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1",
            },
            {
                "filename": "sub-02",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2",
            },
        ]
    },
    "derivatives_freesurfer_sub1": {
        "files": [
            {
                "filename": "ses-mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1_ses_mri",
            },
        ]
    },
    "derivatives_freesurfer_sub1_ses_mri": {
        "files": [
            {
                "filename": "anat",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1_ses_mri_anat",
            },
        ]
    },
    "derivatives_freesurfer_sub1_ses_mri_anat": {
        "files": [
            {
                "filename": "label",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1_ses_mri_anat_label",
            },
            {
                "filename": "mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1_ses_mri_anat_mri",
            },
            {
                "filename": "surf",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub1_ses_mri_anat_surf",
            },
        ]
    },
    "derivatives_freesurfer_sub1_ses_mri_anat_label": {
        "files": [
            {
                "filename": ".lh.BA.thresh.annot.f3h5wZ",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/.lh.BA.thresh.annot.f3h5wZ"
                ],
                "size": 1000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_hidden",
            },
            {
                "filename": "lh.BA.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/lh.BA.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_lh_ba",
            },
            {
                "filename": "lh.BA.thresh.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/lh.BA.thresh.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_lh_ba_thresh",
            },
            {
                "filename": "lh.aparc.DKTatlas40.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/lh.aparc.DKTatlas40.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_lh_aparc_dkt",
            },
            {
                "filename": "lh.aparc.a2009s.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/lh.aparc.a2009s.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_lh_aparc_a2009s",
            },
            {
                "filename": "lh.aparc.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/lh.aparc.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_lh_aparc",
            },
            {
                "filename": "rh.BA.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/rh.BA.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_rh_ba",
            },
            {
                "filename": "rh.BA.thresh.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/rh.BA.thresh.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_rh_ba_thresh",
            },
            {
                "filename": "rh.aparc.DKTatlas40.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/rh.aparc.DKTatlas40.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_rh_aparc_dkt",
            },
            {
                "filename": "rh.aparc.a2009s.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/rh.aparc.a2009s.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_rh_aparc_a2009s",
            },
            {
                "filename": "rh.aparc.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/label/rh.aparc.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_label_rh_aparc",
            },
        ]
    },
    "derivatives_freesurfer_sub1_ses_mri_anat_mri": {
        "files": [
            {
                "filename": "T1.mgz",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/mri/T1.mgz"
                ],
                "size": 50000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_mri_t1",
            },
            {
                "filename": "aseg.mgz",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/mri/aseg.mgz"
                ],
                "size": 10000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_mri_aseg",
            },
        ]
    },
    "derivatives_freesurfer_sub1_ses_mri_anat_surf": {
        "files": [
            {
                "filename": "lh.pial",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/lh.pial"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_lh_pial",
            },
            {
                "filename": "lh.sphere.reg",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/lh.sphere.reg"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_lh_sphere",
            },
            {
                "filename": "lh.white",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/lh.white"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_lh_white",
            },
            {
                "filename": "rh.pial",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/rh.pial"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_rh_pial",
            },
            {
                "filename": "rh.sphere.reg",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/rh.sphere.reg"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_rh_sphere",
            },
            {
                "filename": "rh.white",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-01/ses-mri/anat/surf/rh.white"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub1_surf_rh_white",
            },
        ]
    },
    "derivatives_freesurfer_sub2": {
        "files": [
            {
                "filename": "ses-mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2_ses_mri",
            },
        ]
    },
    "derivatives_freesurfer_sub2_ses_mri": {
        "files": [
            {
                "filename": "anat",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2_ses_mri_anat",
            },
        ]
    },
    "derivatives_freesurfer_sub2_ses_mri_anat": {
        "files": [
            {
                "filename": "label",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2_ses_mri_anat_label",
            },
            {
                "filename": "mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2_ses_mri_anat_mri",
            },
            {
                "filename": "surf",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "derivatives_freesurfer_sub2_ses_mri_anat_surf",
            },
        ]
    },
    "derivatives_freesurfer_sub2_ses_mri_anat_label": {
        "files": [
            {
                "filename": "lh.BA.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/lh.BA.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_lh_ba",
            },
            {
                "filename": "lh.BA.thresh.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/lh.BA.thresh.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_lh_ba_thresh",
            },
            {
                "filename": "lh.aparc.DKTatlas40.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/lh.aparc.DKTatlas40.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_lh_aparc_dkt",
            },
            {
                "filename": "lh.aparc.a2009s.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/lh.aparc.a2009s.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_lh_aparc_a2009s",
            },
            {
                "filename": "lh.aparc.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/lh.aparc.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_lh_aparc",
            },
            {
                "filename": "rh.BA.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/rh.BA.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_rh_ba",
            },
            {
                "filename": "rh.BA.thresh.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/rh.BA.thresh.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_rh_ba_thresh",
            },
            {
                "filename": "rh.aparc.DKTatlas40.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/rh.aparc.DKTatlas40.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_rh_aparc_dkt",
            },
            {
                "filename": "rh.aparc.a2009s.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/rh.aparc.a2009s.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_rh_aparc_a2009s",
            },
            {
                "filename": "rh.aparc.annot",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/label/rh.aparc.annot"
                ],
                "size": 2000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_label_rh_aparc",
            },
        ]
    },
    "derivatives_freesurfer_sub2_ses_mri_anat_mri": {
        "files": [
            {
                "filename": "T1.mgz",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/mri/T1.mgz"
                ],
                "size": 50000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_mri_t1",
            },
            {
                "filename": "aseg.mgz",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/mri/aseg.mgz"
                ],
                "size": 10000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_mri_aseg",
            },
        ]
    },
    "derivatives_freesurfer_sub2_ses_mri_anat_surf": {
        "files": [
            {
                "filename": "lh.pial",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/lh.pial"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_lh_pial",
            },
            {
                "filename": "lh.sphere.reg",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/lh.sphere.reg"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_lh_sphere",
            },
            {
                "filename": "lh.white",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/lh.white"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_lh_white",
            },
            {
                "filename": "rh.pial",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/rh.pial"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_rh_pial",
            },
            {
                "filename": "rh.sphere.reg",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/rh.sphere.reg"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_rh_sphere",
            },
            {
                "filename": "rh.white",
                "urls": [
                    "http://example.com/derivatives/freesurfer/sub-02/ses-mri/anat/surf/rh.white"
                ],
                "size": 30000,
                "directory": False,
                "id": "derivatives_freesurfer_sub2_surf_rh_white",
            },
        ]
    },
    "sub1": {
        "files": [
            {
                "filename": "ses-meg",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_meg",
            },
            {
                "filename": "ses-mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_mri",
            },
        ]
    },
    "sub1_ses_meg": {
        "files": [
            {
                "filename": "sub-01_ses-meg_scans.tsv",
                "urls": ["http://example.com/sub-01/ses-meg/sub-01_ses-meg_scans.tsv"],
                "size": 500,
                "directory": False,
                "id": "sub1_ses_meg_scans",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_channels.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_meg_channels",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_meg.json",
                "urls": [
                    "http://example.com/sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json"
                ],
                "size": 1200,
                "directory": False,
                "id": "sub1_ses_meg_meg_json",
            },
            {
                "filename": "beh",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_meg_beh",
            },
            {
                "filename": "meg",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_meg_meg",
            },
        ]
    },
    "sub1_ses_meg_beh": {
        "files": [
            {
                "filename": "sub-01_ses-meg_task-facerecognition_events.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub1_ses_meg_beh_events",
            },
        ]
    },
    "sub1_ses_meg_meg": {
        "files": [
            {
                "filename": "sub-01_ses-meg_coordsystem.json",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub1_ses_meg_meg_coordsystem",
            },
            {
                "filename": "sub-01_ses-meg_headshape.pos",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos"
                ],
                "size": 15000,
                "directory": False,
                "id": "sub1_ses_meg_meg_headshape",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_meg_meg_run01_events",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub1_ses_meg_meg_run01_fif",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_meg_meg_run02_events",
            },
            {
                "filename": "sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
                "urls": [
                    "http://example.com/sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub1_ses_meg_meg_run02_fif",
            },
        ]
    },
    "sub1_ses_mri": {
        "files": [
            {
                "filename": "anat",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_mri_anat",
            },
            {
                "filename": "dwi",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_mri_dwi",
            },
            {
                "filename": "func",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_mri_func",
            },
            {
                "filename": "fmap",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub1_ses_mri_fmap",
            },
        ]
    },
    "sub1_ses_mri_anat": {
        "files": [
            {
                "filename": "sub-01_ses-mri_acq-mprage_T1w.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_mri_anat_t1w_json",
            },
            {
                "filename": "sub-01_ses-mri_acq-mprage_T1w.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.nii.gz"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_t1w_nii",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo1",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo2",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo3",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo4",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo5",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo6",
            },
            {
                "filename": "sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash1_echo7",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo1",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo2",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo3",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo4",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo5",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo6",
            },
            {
                "filename": "sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub1_ses_mri_anat_flash2_echo7",
            },
        ]
    },
    "sub1_ses_mri_dwi": {
        "files": [
            {
                "filename": "sub-01_ses-mri_dwi.bval",
                "urls": [
                    "http://example.com/sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bval"
                ],
                "size": 1000,
                "directory": False,
                "id": "sub1_ses_mri_dwi_bval",
            },
            {
                "filename": "sub-01_ses-mri_dwi.bvec",
                "urls": [
                    "http://example.com/sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bvec"
                ],
                "size": 2000,
                "directory": False,
                "id": "sub1_ses_mri_dwi_bvec",
            },
            {
                "filename": "sub-01_ses-mri_dwi.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_mri_dwi_json",
            },
            {
                "filename": "sub-01_ses-mri_dwi.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.nii.gz"
                ],
                "size": 30000000,
                "directory": False,
                "id": "sub1_ses_mri_dwi_nii",
            },
        ]
    },
    "sub1_ses_mri_fmap": {
        "files": [
            {
                "filename": "sub-01_ses-mri_magnitude1.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub1_ses_mri_fmap_mag1_json",
            },
            {
                "filename": "sub-01_ses-mri_magnitude1.nii",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub1_ses_mri_fmap_mag1_nii",
            },
            {
                "filename": "sub-01_ses-mri_magnitude2.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub1_ses_mri_fmap_mag2_json",
            },
            {
                "filename": "sub-01_ses-mri_magnitude2.nii",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub1_ses_mri_fmap_mag2_nii",
            },
            {
                "filename": "sub-01_ses-mri_phasediff.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub1_ses_mri_fmap_phasediff_json",
            },
            {
                "filename": "sub-01_ses-mri_phasediff.nii",
                "urls": [
                    "http://example.com/sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub1_ses_mri_fmap_phasediff_nii",
            },
        ]
    },
    "sub1_ses_mri_func": {
        "files": [
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-01_bold.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_mri_func_run01_bold_json",
            },
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz"
                ],
                "size": 40000000,
                "directory": False,
                "id": "sub1_ses_mri_func_run01_bold_nii",
            },
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-01_events.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub1_ses_mri_func_run01_events",
            },
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-02_bold.json",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub1_ses_mri_func_run02_bold_json",
            },
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz"
                ],
                "size": 40000000,
                "directory": False,
                "id": "sub1_ses_mri_func_run02_bold_nii",
            },
            {
                "filename": "sub-01_ses-mri_task-facerecognition_run-02_events.tsv",
                "urls": [
                    "http://example.com/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub1_ses_mri_func_run02_events",
            },
        ]
    },
    "sub2": {
        "files": [
            {
                "filename": "ses-meg",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_meg",
            },
            {
                "filename": "ses-mri",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_mri",
            },
        ]
    },
    "sub2_ses_meg": {
        "files": [
            {
                "filename": "sub-02_ses-meg_scans.tsv",
                "urls": ["http://example.com/sub-02/ses-meg/sub-02_ses-meg_scans.tsv"],
                "size": 500,
                "directory": False,
                "id": "sub2_ses_meg_scans",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_channels.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-meg/sub-02_ses-meg_task-facerecognition_channels.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_meg_channels",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_meg.json",
                "urls": [
                    "http://example.com/sub-02/ses-meg/sub-02_ses-meg_task-facerecognition_meg.json"
                ],
                "size": 1200,
                "directory": False,
                "id": "sub2_ses_meg_meg_json",
            },
            {
                "filename": "beh",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_meg_beh",
            },
            {
                "filename": "meg",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_meg_meg",
            },
        ]
    },
    "sub2_ses_meg_beh": {
        "files": [
            {
                "filename": "sub-02_ses-meg_task-facerecognition_events.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-meg/beh/sub-02_ses-meg_task-facerecognition_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub2_ses_meg_beh_events",
            },
        ]
    },
    "sub2_ses_meg_meg": {
        "files": [
            {
                "filename": "sub-02_ses-meg_coordsystem.json",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_coordsystem.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub2_ses_meg_meg_coordsystem",
            },
            {
                "filename": "sub-02_ses-meg_headshape.pos",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_headshape.pos"
                ],
                "size": 15000,
                "directory": False,
                "id": "sub2_ses_meg_meg_headshape",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_run-01_events.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_events.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_meg_meg_run01_events",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_run-01_meg.fif",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_meg.fif"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub2_ses_meg_meg_run01_fif",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_run-02_events.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-02_events.tsv"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_meg_meg_run02_events",
            },
            {
                "filename": "sub-02_ses-meg_task-facerecognition_run-02_meg.fif",
                "urls": [
                    "http://example.com/sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-02_meg.fif"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub2_ses_meg_meg_run02_fif",
            },
        ]
    },
    "sub2_ses_mri": {
        "files": [
            {
                "filename": "anat",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_mri_anat",
            },
            {
                "filename": "dwi",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_mri_dwi",
            },
            {
                "filename": "func",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_mri_func",
            },
            {
                "filename": "fmap",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub2_ses_mri_fmap",
            },
        ]
    },
    "sub2_ses_mri_anat": {
        "files": [
            {
                "filename": "sub-02_ses-mri_acq-mprage_T1w.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_acq-mprage_T1w.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_mri_anat_t1w_json",
            },
            {
                "filename": "sub-02_ses-mri_acq-mprage_T1w.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_acq-mprage_T1w.nii.gz"
                ],
                "size": 50000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_t1w_nii",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-1_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-1_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo1",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-2_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-2_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo2",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-3_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-3_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo3",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-4_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-4_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo4",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-5_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-5_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo5",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-6_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-6_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo6",
            },
            {
                "filename": "sub-02_ses-mri_run-1_echo-7_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-1_echo-7_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash1_echo7",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-1_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-1_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo1",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-2_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-2_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo2",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-3_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-3_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo3",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-4_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-4_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo4",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-5_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-5_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo5",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-6_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-6_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo6",
            },
            {
                "filename": "sub-02_ses-mri_run-2_echo-7_FLASH.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/anat/sub-02_ses-mri_run-2_echo-7_FLASH.nii.gz"
                ],
                "size": 20000000,
                "directory": False,
                "id": "sub2_ses_mri_anat_flash2_echo7",
            },
        ]
    },
    "sub2_ses_mri_dwi": {
        "files": [
            {
                "filename": "sub-02_ses-mri_dwi.bval",
                "urls": [
                    "http://example.com/sub-02/ses-mri/dwi/sub-02_ses-mri_dwi.bval"
                ],
                "size": 1000,
                "directory": False,
                "id": "sub2_ses_mri_dwi_bval",
            },
            {
                "filename": "sub-02_ses-mri_dwi.bvec",
                "urls": [
                    "http://example.com/sub-02/ses-mri/dwi/sub-02_ses-mri_dwi.bvec"
                ],
                "size": 2000,
                "directory": False,
                "id": "sub2_ses_mri_dwi_bvec",
            },
            {
                "filename": "sub-02_ses-mri_dwi.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/dwi/sub-02_ses-mri_dwi.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_mri_dwi_json",
            },
            {
                "filename": "sub-02_ses-mri_dwi.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/dwi/sub-02_ses-mri_dwi.nii.gz"
                ],
                "size": 30000000,
                "directory": False,
                "id": "sub2_ses_mri_dwi_nii",
            },
        ]
    },
    "sub2_ses_mri_fmap": {
        "files": [
            {
                "filename": "sub-02_ses-mri_magnitude1.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_magnitude1.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub2_ses_mri_fmap_mag1_json",
            },
            {
                "filename": "sub-02_ses-mri_magnitude1.nii",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_magnitude1.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub2_ses_mri_fmap_mag1_nii",
            },
            {
                "filename": "sub-02_ses-mri_magnitude2.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_magnitude2.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub2_ses_mri_fmap_mag2_json",
            },
            {
                "filename": "sub-02_ses-mri_magnitude2.nii",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_magnitude2.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub2_ses_mri_fmap_mag2_nii",
            },
            {
                "filename": "sub-02_ses-mri_phasediff.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_phasediff.json"
                ],
                "size": 400,
                "directory": False,
                "id": "sub2_ses_mri_fmap_phasediff_json",
            },
            {
                "filename": "sub-02_ses-mri_phasediff.nii",
                "urls": [
                    "http://example.com/sub-02/ses-mri/fmap/sub-02_ses-mri_phasediff.nii"
                ],
                "size": 10000000,
                "directory": False,
                "id": "sub2_ses_mri_fmap_phasediff_nii",
            },
        ]
    },
    "sub2_ses_mri_func": {
        "files": [
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-01_bold.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-01_bold.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_mri_func_run01_bold_json",
            },
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-01_bold.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-01_bold.nii.gz"
                ],
                "size": 40000000,
                "directory": False,
                "id": "sub2_ses_mri_func_run01_bold_nii",
            },
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-01_events.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-01_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub2_ses_mri_func_run01_events",
            },
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-02_bold.json",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-02_bold.json"
                ],
                "size": 800,
                "directory": False,
                "id": "sub2_ses_mri_func_run02_bold_json",
            },
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-02_bold.nii.gz",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-02_bold.nii.gz"
                ],
                "size": 40000000,
                "directory": False,
                "id": "sub2_ses_mri_func_run02_bold_nii",
            },
            {
                "filename": "sub-02_ses-mri_task-facerecognition_run-02_events.tsv",
                "urls": [
                    "http://example.com/sub-02/ses-mri/func/sub-02_ses-mri_task-facerecognition_run-02_events.tsv"
                ],
                "size": 600,
                "directory": False,
                "id": "sub2_ses_mri_func_run02_events",
            },
        ]
    },
    "sub-emptyroom": {
        "files": [
            {
                "filename": "ses-20090409",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub-emptyroom_ses-20090409",
            },
        ]
    },
    "sub-emptyroom_ses-20090409": {
        "files": [
            {
                "filename": "sub-emptyroom_ses-20090409_scans.tsv",
                "urls": [
                    "http://example.com/sub-emptyroom/ses-20090409/sub-emptyroom_ses-20090409_scans.tsv"
                ],
                "size": 123,
                "directory": False,
                "id": "sub-emptyroom_ses-20090409_scans",
            },
            {
                "filename": "meg",
                "urls": [],
                "size": 0,
                "directory": True,
                "id": "sub-emptyroom_ses-20090409_meg",
            },
        ]
    },
    "sub-emptyroom_ses-20090409_meg": {
        "files": [
            {
                "filename": "sub-emptyroom_ses-20090409_task-noise_meg.fif",
                "urls": [
                    "http://example.com/sub-emptyroom/ses-20090409/meg/sub-emptyroom_ses-20090409_task-noise_meg.fif"
                ],
                "size": 456,
                "directory": False,
                "id": "sub-emptyroom_ses-20090409_task-noise_meg",
            },
        ]
    },
}


@pytest.mark.parametrize(
    ("dataset", "include", "expected_files"),
    [
        # Test with a single file in root level
        (
            "ds000117",
            ["*.tsv"],
            [
                "dataset_description.json",
                "participants.tsv",
                "participants.json",
                "README",
                "CHANGES",
            ],
        ),
        # Test sub-01
        (
            "ds000117",
            ["sub-01"],
            [
                "CHANGES",
                "README",
                "dataset_description.json",
                "participants.json",
                "participants.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_scans.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json",
                "sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.json",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bval",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bvec",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.json",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.nii.gz",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.nii",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_events.tsv",
            ],
        ),
        # Test sub-01/
        (
            "ds000117",
            ["sub-01/"],
            [
                "CHANGES",
                "README",
                "dataset_description.json",
                "participants.json",
                "participants.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_scans.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json",
                "sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.json",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bval",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bvec",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.json",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.nii.gz",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.nii",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_events.tsv",
            ],
        ),
        # Test sub-01/**
        (
            "ds000117",
            ["sub-01/**"],
            [
                "CHANGES",
                "README",
                "dataset_description.json",
                "participants.json",
                "participants.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_scans.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json",
                "sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.json",
                "sub-01/ses-mri/anat/sub-01_ses-mri_acq-mprage_T1w.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-1_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-1_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-2_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-3_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-4_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-5_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-6_FLASH.nii.gz",
                "sub-01/ses-mri/anat/sub-01_ses-mri_run-2_echo-7_FLASH.nii.gz",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bval",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.bvec",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.json",
                "sub-01/ses-mri/dwi/sub-01_ses-mri_dwi.nii.gz",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude1.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_magnitude2.nii",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.json",
                "sub-01/ses-mri/fmap/sub-01_ses-mri_phasediff.nii",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.json",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_bold.nii.gz",
                "sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-02_events.tsv",
            ],
        ),
        # Test multiple include patterns
        (
            "ds000117",
            [
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_*",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_*",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/*.tsv",
                "sub-01/ses-meg/*.json",
                "sub-emptyroom/ses-20090409",
                "derivatives/meg_derivatives/ct_sparse.fif",
                "derivatives/meg_derivatives/sss_cal.dat",
            ],
            [
                "CHANGES",
                "README",
                "dataset_description.json",
                "participants.json",
                "participants.tsv",
                "derivatives/meg_derivatives/ct_sparse.fif",
                "derivatives/meg_derivatives/sss_cal.dat",
                "sub-01/ses-meg/sub-01_ses-meg_scans.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_channels.tsv",
                "sub-01/ses-meg/sub-01_ses-meg_task-facerecognition_meg.json",
                "sub-01/ses-meg/beh/sub-01_ses-meg_task-facerecognition_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_coordsystem.json",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
                "sub-emptyroom/ses-20090409/sub-emptyroom_ses-20090409_scans.tsv",
                "sub-emptyroom/ses-20090409/meg/sub-emptyroom_ses-20090409_task-noise_meg.fif",
            ],
        ),
    ],
)
def test_download_file_list_generation(
    dataset: str, include: list[str], expected_files: list[str]
):
    """Test that the download function generates the correct list of files without actually downloading.

    This test verifies the file filtering logic by mocking the metadata retrieval
    and checking that the correct files are selected based on include/exclude patterns.
    """

    def mock_get_download_metadata(*args, **kwargs):
        tree = kwargs.get("tree", "null").strip('"').strip("'")
        return copy.deepcopy(MOCK_METADATA[tree])

    def mock_get_local_tag(*args, **kwargs):
        return None

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments"""
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
    [
        ("ds000117", ["*"], 2626),
        ("ds000117", ["sub-01"], 76),
        ("ds000117", ["sub-01/**/*.tsv"], 23),
        ("ds000117", ["sub-01/**"], 76),
        (
            "ds000117",
            [
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_*",
                "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_*",
                "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
                "sub-01/ses-meg/*.tsv",
                "sub-01/ses-meg/*.json",
                "sub-emptyroom/ses-20090409",
                "derivatives/meg_derivatives/ct_sparse.fif",
                "derivatives/meg_derivatives/sss_cal.dat",
            ],
            23,
        ),
        ("ds000117", ["**/ses-meg/**"], 517),
    ],
)
def test_download_file_count(dataset: str, include: list[str], expected_num_files: int):
    """
        Test that the download function generates the correct
        number of files without actually downloading.
    
        This test verifies the file filtering logic by mocking
        the metadata retrieval and checking that the correct
        number of files are selected based on include patterns.
    """

    async def _download_files_spy(*, files, **kwargs):
        """Spy on _download_files to capture the call arguments"""
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
