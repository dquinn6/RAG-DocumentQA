import yaml
import logging

# Load the config file

def load_config_yml(filename):
    try:
        with open(filename, 'rt') as f:
            config = yaml.safe_load(f.read())
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}")
    
    return config

# store user config as var
filepath = "config/user_config.yml"
user_config = load_config_yml(filepath)