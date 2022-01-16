import os
import shutil 
import math
import random
import yaml
import glob

from . import trainvaltest

def gen_crossval(kfolds=1):
    '''
    Generate 
    Sample_command: python -m pre split --whichos linux --path /home/user/path_to_data_config.yaml --modalities ['dc','pc','ec']
    '''
    trainvaltest.ppp('Yay')
    return

def combine_train_test_val(path):
    '''
    For a given modality combines the subjects of train val test folders and stores in a file
    Args:
        path: path of the modality whose children folders are train val test
    '''
    paths = {'AML':list(), 'CCRCC':list()}
    all_classes_path = glob.glob(path+'/*/*/*')

    for subject in all_classes_path:
        subjectCopy = subject[:]
        clas = subjectCopy.rsplit(os.path.sep, 2)[-2]
        paths[clas].append(subject)

    with open(os.path.join(path,'allSubjectPaths.yaml'),'w') as file:
        yaml.dump(paths,file)
    
             