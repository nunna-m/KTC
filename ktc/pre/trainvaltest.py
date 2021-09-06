import os
import sys
import shutil 
import math
import random
from pathlib import Path
import platform
import yaml
import ast


def split(whichos, path, modalities):
    '''
    Program to generate train val test split folders
    Sample_command: python -m pre split --os linux --path /home/user/path_to_data_config.yaml --modalities ['dc','pc','ec']

    Args:
        whichos: linux, windows or remote
        path: absolute path to configfile that has information about datapath and targetpath. Currently only supports yaml extension
        modalities: give a list of slice types from ['am','tm','dc','ec','pc']
    '''

    
    if isinstance(modalities, str):
        modalities = modalities.strip('"').strip('[').strip(']').split(',')
    configs = return_configs_asdict(whichos, path)
    dataset_path = configs['os'][whichos]['data_root']
    target_path = configs['os'][whichos]['after_split_path']
    trainvaltest_percentages = configs['trainvaltest_percentages']

    classes = ["AML", "CCRCC"]

    counts = {
        classes[0]: {i:0 for i in modalities},
        classes[1]: {i:0 for i in modalities}
    }

    new_path = os.path.join(target_path, '_'.join(modalities))
    os.makedirs(new_path, exist_ok=True)

    classwise_subjects = dict()
    for clas in classes:
        classwise_subjects[clas] = os.listdir(os.path.join(dataset_path,clas))
        counts, classwise_subjects[clas] = get_slicetypecount_subjects(dataset_path, modalities, counts, clas, classwise_subjects[clas])
        create_train_val_test_folders(new_path,clas,classwise_subjects[clas],trainvaltest_percentages)

    print("done generating {} split folders".format('_'.join(modalities)))
    return 

def return_configs_asdict(whichos,configfilepath):
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
    return config


def get_slicetypecount_subjects(dataset_path,modalities, counts, clas, folders):
    subject_list = []
    for folder in folders:
        folder_path = os.path.join(dataset_path,clas,folder)
        modes = os.listdir(folder_path)
        count=0
        for m in modes:
            if m in modalities:
                counts[clas][m] += 1
                count+=1
        if count==len(modalities):
            subject_list.append(folder_path)
    return counts, subject_list

def create_train_val_test_folders(path, clas, subjects, fractions):
    '''
    path: where to store the train val test split folders
    clas: AML or CCRCC
    subjects: the paths of subject folders in that class
    fractions (dict): train percentage and validation percentage
    '''
    train_subjects, val_subjects, test_subjects = get_train_val_test_subjects(fractions['train'], fractions['val'], len(subjects), subjects)
    for subset in ['train','val', 'test']:
        newPath = os.path.join(path,subset,clas)
        os.makedirs(newPath, exist_ok=True)
        if subset=='train':
            send_subjects = train_subjects
        elif subset == 'val':
            send_subjects = val_subjects
        elif subset == 'test':
            send_subjects = test_subjects
        copysubjects(send_subjects,newPath)

def get_train_val_test_subjects(_train, _val,class_length,subjects):
    train_length,val_length = get_train_val_test_numbers(_train,_val,class_length)
    random.shuffle(subjects)
    return (subjects[:train_length],
            subjects[train_length:train_length+val_length],
            subjects[train_length+val_length:])

def get_train_val_test_numbers(frac_train,frac_val, length):
    train_length = math.floor(frac_train*length)
    val_length = math.floor(frac_val*length)
    return (train_length, val_length)            

def copysubjects(subjects, path):
    for subject in subjects:
        ID = subject.split(os.path.sep)[-1]
        shutil.copytree(subject,os.path.join(path,ID))