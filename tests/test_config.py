"""Test the openneuro-py configuration file."""

from pathlib import Path
from unittest import mock

import pytest

import openneuro
import openneuro._config
from openneuro._config import (
    SOURCE_ENV_VAR,
    Config,
    get_source,
    get_token,
    init_config,
    load_config,
)


def test_config(tmp_path: Path):
    """Test creating and reading the config file."""
    with mock.patch.object(openneuro._config, "CONFIG_PATH", tmp_path / ".openneuro"):
        assert not openneuro._config.CONFIG_PATH.exists()

        with mock.patch("getpass.getpass", lambda _: "test"):
            init_config()
        assert openneuro._config.CONFIG_PATH.exists()

        expected_config = Config(endpoint="https://openneuro.org/", apikey="test")
        assert load_config() == expected_config
        assert get_token() == "test"


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [
        # Nothing set anywhere.
        (None, None, "openneuro"),
        # The environment variable supplies the default...
        (None, "nemar", "nemar"),
        (None, "openneuro", "openneuro"),
        # ...but an explicit argument always wins over it.
        ("openneuro", "nemar", "openneuro"),
        ("nemar", "openneuro", "nemar"),
        # Blank/whitespace env vars are treated as unset.
        (None, "", "openneuro"),
        (None, "   ", "openneuro"),
    ],
)
def test_get_source(
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    env: str | None,
    expected: str,
):
    """Resolve the download source from the argument, then the environment."""
    if env is None:
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(SOURCE_ENV_VAR, env)
    assert get_source(explicit) == expected  # type: ignore[arg-type]


def test_get_source_rejects_unknown_argument(monkeypatch: pytest.MonkeyPatch):
    """An unknown explicit source names the valid choices."""
    monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="The requested source must be one of"):
        get_source("s3")  # type: ignore[arg-type]


def test_get_source_rejects_unknown_env_var(monkeypatch: pytest.MonkeyPatch):
    """An unknown environment value blames the environment, not the caller."""
    monkeypatch.setenv(SOURCE_ENV_VAR, "s3")
    with pytest.raises(ValueError, match=f"The {SOURCE_ENV_VAR} environment variable"):
        get_source()
