import getpass
import json
import os
import stat
from pathlib import Path
from typing import Literal, TypedDict, get_args

import platformdirs

from openneuro._console import cprint

CONFIG_DIR = Path(
    platformdirs.user_config_dir(appname="openneuro-py", appauthor=False, roaming=True)
)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
BASE_URL = "https://openneuro.org/"

#: Where to fetch dataset files from. See `openneuro.download`.
Source = Literal["openneuro", "nemar"]
SOURCES: tuple[Source, ...] = get_args(Source)
DEFAULT_SOURCE: Source = "openneuro"
#: Environment variable that supplies the default when `source` is not passed.
SOURCE_ENV_VAR = "OPENNEURO_SOURCE"


def get_source(source: Source | None = None) -> Source:
    """Resolve which source to download from.

    An explicit `source` argument wins; otherwise the `OPENNEURO_SOURCE`
    environment variable is consulted, falling back to `"openneuro"`.

    Parameters
    ----------
    source
        The explicitly requested source, or `None` to consult the environment.

    Returns
    -------
    The resolved source name.

    Raises
    ------
    ValueError
        When `source` (or the environment variable) is not a known source.

    """
    if source is None:
        candidate = os.getenv(SOURCE_ENV_VAR, "").strip()
        if not candidate:
            return DEFAULT_SOURCE
        origin = f"The {SOURCE_ENV_VAR} environment variable"
    else:
        candidate = source
        origin = "The requested source"

    for known in SOURCES:
        if candidate == known:
            return known
    raise ValueError(
        f"{origin} must be one of {', '.join(SOURCES)}, but got: {candidate!r}"
    )


class Config(TypedDict):
    """Configuration container."""

    endpoint: str
    apikey: str


def init_config() -> None:
    """Initialize a new OpenNeuro configuration file."""
    cprint(
        "🙏 Please login to your OpenNeuro account and go to: "
        "My Account → Obtain an API Key"
    )
    api_key = getpass.getpass("OpenNeuro API key (input hidden): ")

    config: Config = {
        "endpoint": BASE_URL,
        "apikey": api_key,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)


def load_config() -> dict[str, str]:
    """Load an OpenNeuro configuration file, and return its contents.

    Returns
    -------
    dict
        The configuration options.

    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


def get_token() -> str:
    """Get the OpenNeuro API token if configured with the 'login' command.

    Returns
    -------
    The API token if configured.

    Raises
    ------
    ValueError
        When no token has been configured yet.

    """
    if not CONFIG_PATH.exists():
        raise ValueError(
            "Could not read API token as no openneuro-py configuration "
            'file exists. Run "openneuro login" to generate it.'
        )
    config = load_config()
    if "apikey" not in config:
        raise ValueError(
            "An openneuro-py configuration file was found, but did not "
            'contain an "apikey" entry. Run "openneuro login" to '
            "add such an entry."
        )
    return config["apikey"]
