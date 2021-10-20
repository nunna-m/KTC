'''
utility functions to get data
'''

import os
import json

import yaml

def get_configs(path):
    if isinstance(path, str): return get_single_config([path])
    assert isinstance(path, (tuple, list))
    assert path

    configs = list(map(get_single_config, path))
    base_config = configs[0]
    for additional_config in configs[1:]:
        base_config = add_config(base_config, additional_config)
    return base_config

def get_single_config(path):
    extension = os.path.splitext(path)[1]
    if extension == '.json':
        with open(path, 'r') as fp:
            config = json.load(fp)
    elif extension == '.yaml':
        with open(path, 'r') as fp:
            config = yaml.safe_load(fp)
    else:
        raise NotImplementedError(f'Extension {extension[1:]} not handled')
    return config

def add_config(base, extra):
    #implement this as required
    return