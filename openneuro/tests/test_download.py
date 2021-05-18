import pytest
from openneuro import download


dataset_id_aws = 'ds000246'
tag_aws = '1.0.0'
include_aws = 'sub-0001/anat'

dataset_id_on = 'ds000117'
include_on = 'sub-16/ses-meg'

invalid_tag = 'abcdefg'


@pytest.mark.parametrize(
    ('dataset_id', 'tag', 'include'),
    [
        (dataset_id_aws, tag_aws, include_aws),
        (dataset_id_on, None, include_on)
    ]
)
def test_download(tmp_path, dataset_id, tag, include):
    """Test downloading some files."""
    download(dataset=dataset_id, tag=tag, target_dir=tmp_path, include=include)


def test_download_invalid_tag(tmp_path, dataset_id=dataset_id_aws,
                              invalid_tag=invalid_tag):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match='snapshot.*does not exist'):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)
