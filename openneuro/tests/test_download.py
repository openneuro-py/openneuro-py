import pytest
from openneuro import download


@pytest.fixture
def dataset_id():
    return 'ds000246'


@pytest.fixture
def tag():
    return '1.0.0'


@pytest.fixture
def invalid_tag():
    return 'abcdefg'


@pytest.fixture
def include():
    return 'sub-0001/anat'


def test_download(tmp_path, dataset_id, tag, include):
    """Test downloading some files."""
    download(dataset=dataset_id, tag=tag, target_dir=tmp_path, include=include)


def test_download_invalid_tag(tmp_path, dataset_id, invalid_tag):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match='snapshot.*does not exist'):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)
