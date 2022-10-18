import json
from pathlib import Path

import pytest
from openneuro import download


dataset_id_aws = 'ds000246'
tag_aws = '1.0.0'
include_aws = 'sub-0001/anat'
exclude_aws = []

dataset_id_on = 'ds000117'
tag_on = None
include_on = 'sub-16/ses-meg'
exclude_on = '*.fif'  # save GBs of downloads

invalid_tag = 'abcdefg'


@pytest.mark.parametrize(
    ('dataset_id', 'tag', 'include', 'exclude'),
    [
        (dataset_id_aws, tag_aws, include_aws, exclude_aws),
        (dataset_id_on, tag_on, include_on, exclude_on),
    ]
)
def test_download(tmp_path: Path, dataset_id, tag, include, exclude):
    """Test downloading some files."""
    download(dataset=dataset_id, tag=tag, target_dir=tmp_path, include=include,
             exclude=exclude)


def test_download_invalid_tag(tmp_path: Path, dataset_id=dataset_id_aws,
                              invalid_tag=invalid_tag):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match='snapshot.*does not exist'):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)


def test_resume_download(tmp_path: Path):
    """Test resuming of a dataset download."""
    dataset = 'ds000246'
    tag = '1.0.0'
    include = ['CHANGES']
    download(dataset=dataset, tag=tag, target_dir=tmp_path,
             include=include)

    # Download some more files
    include = ['sub-0001/meg/*.jpg']
    download(dataset=dataset, tag=tag, target_dir=tmp_path,
             include=include)

    # Download from a different revision / tag
    new_tag = '00001'
    include = ['CHANGES']
    with pytest.raises(FileExistsError, match=f'revision {tag} exists'):
        download(dataset=dataset, tag=new_tag, target_dir=tmp_path,
                 include=include)

    # Try to "resume" from a different dataset
    new_dataset = 'ds000117'
    with pytest.raises(RuntimeError,
                       match='existing dataset.*appears to be different'):
        download(dataset=new_dataset, target_dir=tmp_path, include=include)

    # Remove "DatasetDOI" from JSON
    json_path = tmp_path / 'dataset_description.json'
    with json_path.open('r', encoding='utf-8') as f:
        dataset_json = json.load(f)

    del dataset_json['DatasetDOI']
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(dataset_json, f)

    with pytest.raises(RuntimeError,
                       match=r'does not contain "DatasetDOI"'):
        download(dataset=dataset, target_dir=tmp_path)

    # We should be able to resume a download even if "datset_description.jon"
    # is missing
    json_path.unlink()
    include = ['sub-0001/meg/sub-0001_coordsystem.json']
    download(dataset=dataset, tag=tag, target_dir=tmp_path,
             include=include)


def test_ds000248(tmp_path: Path):
    """Test a dataset for that we ship default excludes."""
    dataset = 'ds000248'
    download(
        dataset=dataset,
        include=['participants.tsv'],
        target_dir=tmp_path
    )


def test_doi_handling(tmp_path: Path):
    """Test that we can handle DOIs that start with 'doi:`."""
    dataset = 'ds000248'
    download(
        dataset=dataset,
        include=['participants.tsv'],
        target_dir=tmp_path
    )

    # Now inject a `doi:` prefix into the DOI
    dataset_description_path = tmp_path / 'dataset_description.json'
    dataset_description_text = \
        dataset_description_path.read_text(encoding='utf-8')
    dataset_description = json.loads(dataset_description_text)
    # Make sure we can dumps to get the same thing back (if they change their
    # indent 4->8 for example, we might try to resume our download of the file
    # and things will break in a challenging way)
    dataset_description_rt = json.dumps(dataset_description, indent=4)
    assert dataset_description_text == dataset_description_rt
    # Ensure the dataset doesn't already have the problematic prefix, then add
    assert not dataset_description['DatasetDOI'].startswith('doi:')
    dataset_description['DatasetDOI'] = (
        'doi:' + dataset_description['DatasetDOI']
    )
    dataset_description_path.write_text(
        data=json.dumps(dataset_description, indent=4),
        encoding='utf-8'
    )

    # Try to download again
    download(
        dataset=dataset,
        include=['participants.tsv'],
        target_dir=tmp_path
    )
