import json
from src.config import config
import yaml
import logging
from importlib import reload

LOG_PATH = config.user_config["LOG_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

def update_config_yml(new_config: dict):
    try:
        config_file = config.load_config_yml(config.filepath)

        if any([k not in config_file.keys() for k in new_config.keys()]):
            raise ValueError("Invalid update config; one or more keys not found in original config")
        
        config_file.update(new_config)

        with open("src/config/user_config.yml", "w") as f:
            f.write(yaml.dump(config_file))

        # need to reload config module to update imported config.user_config values
        reload(config)
        
        logging.info(f"{config.filepath} overwritten.")

    except Exception as e:
        logging.error(f"Failed to update config file: {e}")


def update_patterns_json(key = None, val = None, clear_json=False):

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    if clear_json:
        patterns_json = {}

    else:
        if ((key not in [None, ""]) and (val not in [None, ""])):
            patterns_json.update({key: val})

    with open(PATTERNS_FILENAME, "w") as w:
        json.dump(patterns_json, w)
        

def test_communication(communicator):
    try:
        _ = communicator.post_prompt("Hi")
    except Exception as e:
        raise ValueError(f"Communication test failed: {e}")


def manipulate_passages(passages, replace_pattern, verbose=True):
    #return list(map(lambda x: x.replace(replace_pattern[0], replace_pattern[1]), passages))

    manipulated_count = 0
    manipulated_passages = []
    for passage in passages:
        if replace_pattern[0] in passage:
            passage = passage.replace(replace_pattern[0], replace_pattern[1])
            manipulated_count += 1

        manipulated_passages.append(passage)

    if verbose:
        logging.info(f"{manipulated_count} passages manipulated; '{replace_pattern[0]}' -> '{replace_pattern[1]}'")

    return manipulated_passages