"""Script to be run after first setup to create initial placeholder files and directories used in other scripts."""

import os
import json
from src.config import config
from src.utils import create_empty_file

LOG_PATH = config.user_config["LOG_PATH"]
SAVE_PATH = config.user_config["SAVE_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]


if __name__ == "__main__":

    # Init vectorstore dir, log and pattern files with empty placeholders
    for filepath in [
        SAVE_PATH + "__init__.py",
        LOG_PATH + "backend.log", 
        LOG_PATH + "streamlit.log",     
        PATTERNS_FILENAME,
    ]:
        if not os.path.exists(filepath):
            create_empty_file(filepath)

    # Init patterns json with empty dict
    with open(PATTERNS_FILENAME, "w") as w:
        json.dump({}, w)