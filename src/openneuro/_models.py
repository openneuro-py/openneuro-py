"""Pydantic models for validating OpenNeuro GraphQL API responses.

Only the inner payload objects (snapshots, files) are modeled here.
The outer GraphQL response envelope (`{"data": {"dataset": ...}}`) is
traversed with plain dict access in the download module.
"""

from pydantic import BaseModel


class DatasetFile(BaseModel):
    """Metadata for a single file in a dataset snapshot."""

    filename: str
    urls: list[str] | None = None
    size: int | None = None
    id: str


class Snapshot(BaseModel):
    """A dataset snapshot containing an ID and a list of files."""

    id: str
    files: list[DatasetFile]
