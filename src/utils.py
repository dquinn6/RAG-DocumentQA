"""Module containing utility functions used throughout the codebase."""

import json
import logging
import os
from importlib import reload
from typing import List, Optional, Tuple

import yaml

from src.communicators import Communicator, CommunicatorError
from src.config import config

LOG_PATH = config.user_config["LOG_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]


def create_empty_file(filepath: str):
    """Create an empty placeholder file; creates file directory if it doesn't exist.

    Args:
        filepath (str): File path + name to create
    """
    try:
        # Create dir if it doesn't exist
        basedir = os.path.dirname(filepath)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # Create empty file
        with open(filepath, "w+") as f:
            pass

    except Exception as e:
        logging.error(f"Failed to create empty file: {e}")


def update_config_yml(new_config: dict) -> None:
    """Update program config.yml with new values.

    Args:
        new_config (dict): Update dict, where key matches one in current config.yml and value is val to update to.

    Raises:
        ValueError: Raised when new_config contains an unknown key.
    """
    try:
        config_file = config.load_config_yml(config.filepath)

        if any([k not in config_file.keys() for k in new_config.keys()]):
            raise ValueError(
                "Invalid update config; one or more keys not found in original config"
            )

        config_file.update(new_config)

        with open(config.filepath, "w") as f:
            f.write(yaml.dump(config_file))

        # Need to reload config module to update imported config.user_config values
        reload(config)

        logging.info(f"{config.filepath} overwritten.")

    except Exception as e:
        logging.error(f"Failed to update config file: {e}")


def update_patterns_json(
    search_key: Optional[str] = None,
    replace_val: Optional[str] = None,
    clear_json: bool = False,
) -> None:
    """Update the search and replace patterns json with new values.

    Args:
        search_key (Optional[str], optional): Search key value. Defaults to None.
        replace_val (Optional[str], optional): Replace key value. Defaults to None.
        clear_json (bool, optional): If true, clears json to empty dict. Defaults to False.
    """
    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    if clear_json:
        patterns_json = {}

    else:
        if (search_key not in [None, ""]) and (replace_val not in [None, ""]):
            patterns_json.update({search_key: replace_val})

    with open(PATTERNS_FILENAME, "w") as w:
        json.dump(patterns_json, w)


def test_communication(communicator: Communicator) -> None:
    """Test communication with model using a short message.

    Args:
        communicator (Communicator): Communicator object for sending messages to model.

    Raises:
        CommunicatorError: Raised if communication failed.
    """
    try:
        _ = communicator.post_prompt("Hi")
    except Exception as e:
        raise CommunicatorError(f"Communication test failed: {e}")


def manipulate_passages(
    passages: List[str], replace_pattern: Tuple[str, str], verbose: bool = True
) -> Optional[List[str]]:
    """Manipulate list of text with a search and replace pattern. If search pattern is not in test, it is returned unmodified.

    Not necessarily a vital method for the program; will return None instead of raising error in Exception.

    Args:
        passages (List[str]): List of text to manipulate.
        replace_pattern (str): A tuple pair, where the first element is the search pattern and the second element is the replae pattern.
        verbose (bool, optional): Display logging info messages. Defaults to True.

    Returns:
        Optional[List[str]]: List of manipulated passages
    """
    try:
        manipulated_count = 0
        manipulated_passages = []
        for passage in passages:
            if replace_pattern[0] in passage:
                passage = passage.replace(
                    replace_pattern[0], replace_pattern[1]
                )
                manipulated_count += 1

            manipulated_passages.append(passage)

        if verbose:
            logging.info(
                f"{manipulated_count} passages manipulated; '{replace_pattern[0]}' -> '{replace_pattern[1]}'"
            )

    except Exception as e:
        logging.error(f"Failed to manipulate passages: {e}")
        return None

    return manipulated_passages
