from pathlib import Path
import os
import stat
import json
import getpass

from tqdm.auto import tqdm


config_fname = Path('~/.openneuro').expanduser()
default_base_url = 'https://openneuro.org/'


def init_config() -> None:
    """Initialize a new OpenNeuro configuration file.
    """
    tqdm.write('🙏 Please login to your OpenNeuro account and go to: '
               'My Account → Obtain an API Key')
    api_key = getpass.getpass('OpenNeuro API key (input hidden): ')
    config = dict(url=default_base_url,
                  apikey=api_key,
                  errorReporting=False)
    with open(config_fname, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    os.chmod(config_fname, stat.S_IRUSR)


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


def get_token() -> str:
    """Get the OpenNeuro API token if configured with the 'login' command.

    Returns
    -------
    str
        The API token if configured.

    Raises
    ------
    ValueError
        When no token has been configured yet.
    """
    if not config_fname.exists():
        raise ValueError('Could not read API token as no ~/.openneuro config '
                         'file exists. Run "openneuro login" to generate it.')
    config = load_config()
    if 'apikey' not in config:
        raise ValueError('An ~/.openneuro config file was found, but did not '
                         'contain an "apikey" entry. Run "openneuro login" to '
                         'add such an entry.')
    return config['apikey']
