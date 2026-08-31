"""Pydantic models for validating dataset metadata responses.

Only the inner payload objects (snapshots, files) are modeled here.
The outer GraphQL response envelope (`{"data": {"dataset": ...}}`) is
traversed with plain dict access in the download module.

The same models describe files served by the NEMAR mirror (see
`openneuro._nemar`), whose manifest additionally carries a per-file
checksum. OpenNeuro's GraphQL API does not expose one, so the checksum
fields are optional and left unset there.
"""

from typing import Literal

from pydantic import BaseModel

#: Checksum algorithms that may appear in a NEMAR manifest.
#:
#: `"git"` is a git *blob* hash: SHA-1 over ``b"blob <size>\0"`` followed by
#: the file contents, not a plain SHA-1 of the contents.
ChecksumAlgorithm = Literal["md5", "sha256", "git"]


class DatasetFile(BaseModel):
    """Metadata for a single file in a dataset snapshot."""

    filename: str
    urls: list[str] | None = None
    size: int | None = None
    id: str
    checksum: str | None = None
    checksum_algorithm: ChecksumAlgorithm | None = None


class Snapshot(BaseModel):
    """A dataset snapshot containing an ID and a list of files."""

    id: str
    files: list[DatasetFile]
