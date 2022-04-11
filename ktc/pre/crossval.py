from operator import mod
import os
import random
from xgboost import train
import yaml
import glob
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
import numpy as np

from . import trainvaltest

def crossvalDataGen(whichos, path, modalities, kfolds, loocv=False):
    '''
    Outer function to call on combining train val and test subjects. 
    Sample_command: python -m pre crossvalDataGen --path /home/user/path_to_config --modalities /list/of/modalities --kfolds number/of/folds/to/split --whichos operating_sys
    Args:
        whichos: linux/windows/remote
        path: path to the config file that has data paths
        modalities (list[str]): pass a list of modalities to generate files for crossval subjects
        kfolds: number of crossvalidation folds
        loocv (bool): generate Leave one out cross validation data

    '''
    folds = int(kfolds)
    
    if isinstance(modalities, str):
        modalities = modalities.strip('""][').replace("'","").split(',')
    modalities = sorted(modalities, reverse=False)
    print(' '.join(modalities))
    configs = trainvaltest.return_configs_asdict(whichos, path)
    basepath = os.path.join(configs['os'][whichos]['after_split_path'])
    modalityPath = os.path.join(basepath, '_'.join(modalities))
    combine_train_test_val(modalityPath, kfolds=folds, loocv=loocv)
    return

def combine_train_test_val(path, kfolds=1, loocv=False):
    '''
    For a given modality combines the subjects of train val test folders and stores in a file + generates k folds CV data + leave_one_out CV --from all subjects and stores separate files each with train and test subjects for each fold
    Args:
        path: path of the modality whose children folders are train val test
        kfolds: number of folds for cross validation
        loocv: if you want to do Leave one out cross validation also
    '''
    paths = {'AML':list(), 'CCRCC':list()}
    all_classes_path = glob.glob(path+'/*/*/*')
    target_classes = []
    for subject in all_classes_path:
        subjectCopy = subject[:]
        clas = subjectCopy.rsplit(os.path.sep, 2)[-2] #AML or CCRCC
        paths[clas].append(subject)
        target_classes.append(clas)
    
    with open(os.path.join(path,'allSubjectPaths.yaml'),'w') as file:
        yaml.dump(paths,file)

    #k folds CV
    k = KFold(n_splits=kfolds, shuffle=False)
    train_full = {i: list() for i in range(kfolds)}
    test_full = {i: list() for i in range(kfolds)}
    for clas in ['AML', 'CCRCC']:
        fold_num = 0
        for train_index, test_index in k.split(paths[clas]):
            train_full[fold_num].extend(np.take(paths[clas],train_index))
            test_full[fold_num].extend(np.take(paths[clas],test_index))
            fold_num+=1
    storeCVPath = os.path.join(path,'{}CV'.format(kfolds))
    os.makedirs(storeCVPath, exist_ok=True)
    for i in range(kfolds):
        random.shuffle(train_full[i])
        random.shuffle(test_full[i])
        new_train_full = [str(pat) for pat in train_full[i]]
        new_test_full = [str(pat) for pat in test_full[i]]
        store = {'train':new_train_full,
                'test':new_test_full}
        with open(os.path.join(storeCVPath,'allSubjectPaths{}.yaml'.format(i)),'w') as file:
            yaml.dump(store,file)
    
    #leave-one-out CV
    if loocv:
        k = KFold(n_splits=len(paths['AML']), shuffle=False) #taking length of class as it is smaller, doing loocv on AML and CCRCC separately and then joining the lists
        train_full, test_full = dict(), dict()
        for i in range(len(paths['AML'])):
            train_full[i] = list()
            test_full[i] = list()
        for clas in ['AML', 'CCRCC']:
            fold_num = 0
            for train_index, test_index in k.split(paths[clas]):
                train_full[fold_num].extend(np.take(paths[clas],train_index))
                test_full[fold_num].extend(np.take(paths[clas],test_index))
                fold_num+=1
        storeCVPath = os.path.join(path,'LOOCV')
        os.makedirs(storeCVPath, exist_ok=True)
        for i in range(fold_num):
            random.shuffle(train_full[i])
            random.shuffle(test_full[i])
            new_train_full = [str(pat) for pat in train_full[i]]
            new_test_full = [str(pat) for pat in test_full[i]]
            store = {'train':new_train_full,
                    'test':new_test_full}
            with open(os.path.join(storeCVPath,'allSubjectPaths{}.yaml'.format(i)),'w') as file:
                yaml.dump(store,file)