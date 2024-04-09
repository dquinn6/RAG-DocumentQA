"""Module to load values from config.yml and init in module variable for codebase reference."""

import os

import yaml

DEFAULT_CONFIG = {
    "ACCESS_TOKEN": None,
    "MODEL_NAME": "gpt-3.5-turbo",
    "DATASET_NAME": "WikiText",
    "VECTORSTORE_NAME": "LangchainFAISS",
    "LOG_PATH": "logs/",
    "PATTERNS_FILENAME": "src/config/manipulate_patterns.json",
    "SAVE_PATH": "document_store/",
    "SEARCH_TYPE": "similarity",
    "N_RETRIEVED_DOCS": 5,
    "TOKEN_LIMIT": 2000,
    "VERBOSE": True,
}


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

if not os.path.exists(filepath):
    with open(filepath, "w") as f:
        f.write(yaml.dump(DEFAULT_CONFIG))

user_config = load_config_yml(filepath)
