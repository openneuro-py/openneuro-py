"""Shared pytest fixtures for openneuro tests."""

import os
from collections.abc import Iterator

import pytest

import openneuro._console


@pytest.fixture(autouse=True)
def restore_cli_flag() -> Iterator[None]:
    """Keep a test that runs the CLI in-process from leaking its mode."""
    original = openneuro._console._RUNNING_FROM_CLI
    yield
    openneuro._console._RUNNING_FROM_CLI = original


@pytest.fixture(scope="session")
def openneuro_token() -> str | None:
    """Provide OpenNeuro API token from environment."""
    token = os.getenv("OPENNEURO_TEST_TOKEN")
    if not token:
        pytest.skip("OPENNEURO_TEST_TOKEN environment variable not set")
    return token
