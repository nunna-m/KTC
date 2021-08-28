'''
utility functions to get data
'''

import os

import yaml

def get_singleconfigfile_yaml(path):
    '''
    Get Configs
    
    Args:
        path: path to the single config file
    
    Returns:
        config (dictionary usually)
    '''
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config

def add_configs(base, addition):
    '''
    Take the first config as base, and make additions according to the additional configs in the list, passed in as addition here
    '''
    def add_single(previous_base, newkey, newvalue):
        '''
        adding a single key, value pair to the base config
        '''
        previous_base[newkey] = newvalue
    
    for key, value in addition.items():
        base = add_single(base, key, value)
    
    return base

def get_configs_yaml(configfiles):
    '''
    Get configs from specified json file path
    Args:
        json_path: path to json file which contains configs
    Returns:
        config (dict): configs stored in dictionary
    '''
    if isinstance(configfiles, str): return get_configs_yaml([configfiles])
    assert isinstance(configfiles, (tuple, list))
    assert configfiles

    configs = list(map(get_singleconfigfile_yaml,configfiles))
    
    base_config = configs[0]
    for extra in configs[1:]:
        config = add_configs(base_config, extra)
    return config
