"""Utility functions for tests."""

import json
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"


def load_json(path):
    """Load a JSON file."""
    return json.loads((TEST_DATA_DIR / path).read_text("utf-8"))
