import os
import yaml
from ruamel.yaml import YAML


def store_configs(path_save, overwrite=True, **configs):
    '''
    Save configs to a file.

    Args:
        path_save: file path to store the configs
        overwrite (bool): whether you should replace a file that already exists with this one
        configs: configs to store
    Returns:
        None
    '''

    while os.path.exists(path_save):
        filename = os.path.basename(path_save)
        new_filename = '{}_{}'.format(*os.path.splitext(filename))
        

