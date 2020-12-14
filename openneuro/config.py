from pathlib import Path
import json
import getpass


config_fname = Path('~/.openneuro').expanduser()
default_base_url = 'https://openneuro.org/'


def init_config() -> None:
    """Initialize a new OpenNeuro configuration file.
    """
    api_key = getpass.getpass('OpenNeuro API key (input hidden): ')
    config = dict(url=default_base_url,
                  apikey=api_key,
                  errorReporting=False)
    with open(config_fname, 'w', encoding='utf-8') as f:
        json.dump(config, f)


def load_config() -> dict:
    """Load an OpenNeuro configuration file, and return its contents.

    Returns
    -------
    dict
        The configuration options.
    """
    with open(config_fname, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
