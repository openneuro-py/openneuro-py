from pathlib import Path
from unittest import mock

import openneuro
from openneuro.config import init_config, load_config, get_token, Config


def test_config(tmp_path: Path):
    """Test creating and reading the config file."""
    with mock.patch.object(openneuro.config, "CONFIG_PATH", tmp_path / ".openneuro"):
        assert not openneuro.config.CONFIG_PATH.exists()

        with mock.patch("getpass.getpass", lambda _: "test"):
            init_config()
        assert openneuro.config.CONFIG_PATH.exists()

        expected_config = Config(endpoint="https://openneuro.org/", apikey="test")
        assert load_config() == expected_config
        assert get_token() == "test"
