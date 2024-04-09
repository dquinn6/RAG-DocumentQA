"""Module for testing all utility functions."""

import json
import random
import string

from src.config import config
from src.utils import update_config_yml, update_patterns_json

PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]


def test_update_user_config():
    """Test config.yml is properly updated."""

    # Fetch initial config file
    init_config = config.user_config
    init_config_val = init_config["TOKEN_LIMIT"]

    # Define a new random config val to update to
    new_config_val = random.randint(0, 10000)
    # If by chance we get same rand num, get new random number
    while new_config_val == init_config_val:
        new_config_val = random.randint(0, 10000)

    # Overwrite config.yml with new config
    new_config = {
        "TOKEN_LIMIT": new_config_val,
    }
    update_config_yml(new_config)

    # Fetch updated config file and compare with init config
    updated_config = config.user_config

    assert (
        updated_config != init_config
    ), "Failed to properly update config.yml"


def test_update_patterns_config():
    """Test manipulate patterns json is properly updated."""

    # Clear file
    update_patterns_json(clear_json=True)

    # Fetch current patterns saved
    with open(PATTERNS_FILENAME) as f:
        init_patterns = json.load(f)

    # File should have been cleared
    assert init_patterns == {}, "Failed to properly clear patterns.json"

    # Create random patterns to update to
    random_key = "".join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(25)
    )
    random_val = "".join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(25)
    )

    # Overwrite patterns json
    update_patterns_json(random_key, random_val)

    # Fetch updated patterns file and compare with init
    with open(PATTERNS_FILENAME) as f:
        updated_patterns = json.load(f)

    assert (
        init_patterns != updated_patterns
    ), "Failed to properly update patterns.json"
