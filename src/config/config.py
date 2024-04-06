"""Module to load values from config.yml and init in module variable for codebase reference."""

import yaml


class ConfigLoadError(Exception):
    """Custom error for config loading function."""

    pass


def load_config_yml(filepath: str) -> dict:
    """Load config.yml into variable.

    Args:
        filepath (str): Path to config.yml to load.

    Raises:
        ConfigLoadError: Raised when failing to load yml file.

    Returns:
        dict: Loaded config.yml into dict object.
    """
    try:
        with open(filepath, "rt") as f:
            config = yaml.safe_load(f.read())
    except Exception as e:
        raise ConfigLoadError(f"Failed to load config file: {e}")

    return config


# Store user config as var
filepath = "src/config/user_config.yml"
user_config = load_config_yml(filepath)
