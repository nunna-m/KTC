import os
import shutil 
import math
import random
import yaml

def split(whichos, path, modalities):
    '''
    Program to generate train val test split folders
    Sample_command: python -m pre split --whichos linux --path /home/user/path_to_data_config.yaml --modalities ['dc','pc','ec']

    Args:
        whichos: linux,s windows or remote
        path: absolute path to configfile that has information about datapath and targetpath. Currently only supports yaml extension
        modalities: give a list of slice types from ['am','tm','dc','ec','pc']
    '''

    if isinstance(modalities, str):
        modalities = modalities.strip('][').replace("'","").split(',')
    
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
        create_train_val_test_folders(new_path,clas,classwise_subjects[clas],trainvaltest_percentages, modalities)

    print("done generating {} split folders".format('_'.join(modalities)))
    return 

def return_configs_asdict(whichos,configfilepath):
    '''
    Based on OS and configuration file path, configs are returned as a dictionary

    Args:
        whichos: windows, linux or remote
        configfilepath: absolute path to the configuration file that has os, datapath, targetpath, trainvaltestpercentages
    '''
    if whichos.lower() not in ['windows','linux','remote']:
        raise NotImplementedError(f'OS {whichos} option not supported')
    if isinstance(configfilepath,str):
        extension = configfilepath.split(os.sep)[-1].split('.')[-1]
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
    '''
    Returns number of examples for each slice type (modality) and paths to subject folders which have all the specified modalities

    Args:
        dataset_path: absolute path to the full data
        modalities: subset of all modalites, using this subjects are filtered and then further divided into train.val.test
        counts: dictionary of counts
        clas: classification categories
        folders: all the subject folders (pre filtered)
    '''
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

def create_train_val_test_folders(path, clas, subjects, fractions, modalities):
    '''
    Create train val test folders classification category wise from the whole dataset based on modalities specified in cmd

    Args:
        path: where to store the train val test split folders
        clas: classification category
        subjects: the paths of subject folders in that class
        fractions (dict): train percentage and validation percentage, passed in through intial configuration file
        modalities: required modalities for copying
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
        copysubjects(send_subjects,newPath,modalities)

def get_train_val_test_subjects(_train, _val,class_length,subjects):
    '''
    Based on train.val.test percentages split subject folders randomly into train val and test folders

    Args:
        _train: fraction of train
        _val: fraction on validation
        class_length: number of subjects of a specific classification category
        subjects: the subjects' folder paths of a specific classification category
    '''
    train_length,val_length = get_train_val_test_numbers(_train,_val,class_length)
    random.shuffle(subjects)
    return (subjects[:train_length],
            subjects[train_length:train_length+val_length],
            subjects[train_length+val_length:])

def get_train_val_test_numbers(frac_train,frac_val, length):
    '''
    Return number of training and validation subjects based on train and val fractions in configfile
    
    Args:
        frac_train: fraction of training subjects
        frac_val: fraction of val subjects
        length: number of subjects in a specific classification category
    '''
    train_length = math.floor(frac_train*length)
    val_length = math.floor(frac_val*length)
    return (train_length, val_length)            

def copysubjects(subjects, path, modalities):
    '''
    Deep copy of folders from full data to train.val.test split folders
    
    Args:
        subjects: paths of subject folders to perform deep copy (basically source)
        path: target path to store the deep copied folders (basically destination)
        modalities: required modalities for copying
    '''
    modalities_l = [i+'L' for i in modalities]
    modalities_l.extend(modalities)
    #print(path)
    for subject in subjects:
        ID = subject.split(os.path.sep)[-1]
        #print("Subject: {} and ID: {}".format(subject,ID))
        for base, dirs, files in os.walk(subject):
            if len(dirs)>0:
                for dir in dirs:
                    if dir in modalities_l:
                        old_path = os.path.join(subject, dir)
                        new_path = os.path.join(path, ID, dir)
                        shutil.copytree(old_path,new_path)

def remove_existing_folder(whichos, path):
    '''
    Removes existing target path folder after trainvaltest split
    
    Args:
        whichos: linux,s windows or remote
        path: absolute path to configfile that has information about datapath and targetpath. Currently only supports yaml extension
    '''
    configs = return_configs_asdict(whichos, path)
    target_path = configs['os'][whichos]['after_split_path']
    if os.path.exists(target_path) and os.path.isdir(target_path):
        shutil.rmtree(target_path)