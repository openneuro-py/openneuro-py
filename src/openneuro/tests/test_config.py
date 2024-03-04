"""Test the openneuro-py configuration file."""

from pathlib import Path
from unittest import mock

import openneuro
import openneuro._config
from openneuro._config import Config, get_token, init_config, load_config


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
