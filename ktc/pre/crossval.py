import os
import shutil 
import math
import random
import yaml

from . import trainvaltest

def doThis():
    '''
    Sample testing program to generate train val test split folders
    Sample_command: python -m pre split --whichos linux --path /home/user/path_to_data_config.yaml --modalities ['dc','pc','ec']
    '''
    trainvaltest.ppp('Yay')
    return