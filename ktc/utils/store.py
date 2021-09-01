import os
from datetime import datetime
import yaml
import json
from ruamel.yaml import YAML


def store_configs(path_save, **configs):
    '''
    Save configs to a file.

    Args:
        path_save: file path to store the configs
        overwrite (bool): whether you should replace a file that already exists with this one
        configs: configs to store
    Returns:
        None
    '''
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M')
    dirname = os.path.dirname(path_save)
    filename, extension = os.path.splitext(path_save)
    if not os.path.exists(path_save):
        os.makedirs(dirname, exist_ok=True)
    
    path_save = filename+now+extension
    
    
    if extension == '.json':
        with open(path_save,'w') as fp:
            json.dump(configs, fp)
    elif extension == '.yaml':
        with open(path_save,'w') as fp:
            YAML(typ='safe').dump(configs, fp)
    else:
        raise NotImplementedError(f'Pass in the correct file extension: {extension[1:]}')
    
    return

# config = dict()
# config['a']='123'
# config['b']='456'
# fname = 'options.pkl'
# p = '/home/maanvi/LAB/Datasets/kidney_dcecpc_TVT/results/'
# store_configs(
#     os.path.join(p,fname),
#     config=config,
#     )


