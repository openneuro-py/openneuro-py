import pytest
from openneuro import download


@pytest.fixture
def dataset_id():
    return 'ds000246'


@pytest.fixture
def tag():
    return '1.0.0'


@pytest.fixture
def include():
    return 'sub-0001/anat'


def test_download(tmp_path, dataset_id, tag, include):
    """Test downloading some files."""
    download(dataset=dataset_id, tag=tag, target_dir=tmp_path, include=include)
