"""Utility functions for tests."""

import os
import json
from pathlib import Path

TEST_DATA_DIR= os.path.join(Path(__file__).parent, "data")

def load_json(path):
    """Load a JSON file."""
    path = os.path.join(TEST_DATA_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

