"""Shared pytest fixtures for openneuro tests."""

import os

import pytest


@pytest.fixture(scope="session")
def openneuro_token() -> str:
    """Provide OpenNeuro API token from environment."""
    token = os.getenv("OPENNEURO_TEST_TOKEN")
    if not token:
        pytest.skip("OPENNEURO_TEST_TOKEN environment variable not set")
    return token
