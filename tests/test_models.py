"""Tests for openneuro._models Pydantic response models."""

from openneuro._models import DatasetFile, Snapshot
from tests.utils import load_json


def test_dataset_file():
    """Parse a complete DatasetFile payload."""
    fi = DatasetFile.model_validate(
        {
            "filename": "sub-01/anat/T1w.nii.gz",
            "urls": ["https://example.com/file"],
            "size": 1234,
            "id": "abc123",
        }
    )
    assert fi.filename == "sub-01/anat/T1w.nii.gz"
    assert fi.urls == ["https://example.com/file"]
    assert fi.size == 1234
    assert fi.id == "abc123"

    # Null size (GraphQL `size: null`)
    fi_null = DatasetFile.model_validate(
        {
            "filename": "sub-01/anat/T1w.nii.gz",
            "urls": None,
            "size": None,
            "id": "abc123",
        }
    )
    assert fi_null.size is None
    assert fi_null.urls is None


def test_snapshot_from_mock_data():
    """Parse real mock metadata into a Snapshot."""
    data = load_json("mock_metadata_ds000117.json")
    snap = Snapshot.model_validate(data)
    assert snap.id == "ds000117:1.1.0"
    assert len(snap.files) > 0
    assert all(isinstance(f, DatasetFile) for f in snap.files)
