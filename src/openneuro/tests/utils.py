"""Utility functions for tests."""

import json
from pathlib import Path
from typing import Any

TEST_DATA_DIR = Path(__file__).parent / "data"


def load_json(path: str) -> list[Any] | dict[str, Any]:
    """Load a JSON file."""
    out = json.loads((TEST_DATA_DIR / path).read_text("utf-8"))
    assert isinstance(out, (list, dict))
    return out
