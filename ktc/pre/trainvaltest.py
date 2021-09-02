import os
import sys
import shutil 
import math
import random
from pathlib import Path
import platform
import yaml

def generate_trainvaltest_split(whichos,configfilepath):
    if whichos.lower() not in ['windows','linux','remote']:
        raise NotImplementedError(f'OS {whichos} option not supported')
    if isinstance(configfilepath,str):
        extension = os.path.splitext(configfilepath)[1][1:]
        if extension=='yaml' or extension=='yml':
            pass
        else:
            raise NotImplementedError(f'Please pass config file in {extension} format')
    else:
        raise TypeError(f'Pass in data dir as string')

    stream = open(configfilepath, 'r')
    config = yaml.safe_load(stream)
    data_root = config[whichos]['data_root']
    save_path = config[whichos]['after_split_path']
    
    return (data_root, save_path)

def main():
    args = sys.argv[1:]
    os = str(args[0])
    path = str(args[1])

    dataset_path, target_path = generate_trainvaltest_split(os, path)
