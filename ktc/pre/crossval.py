from operator import mod
import os
import random
import yaml
import glob
from sklearn.model_selection import KFold
import numpy as np

from . import trainvaltest

def crossvalDataGen(whichos, path, modalities, kfolds, holdout):
    '''
    Outer function to call on combining train val and test subjects. 
    Sample_command: python -m pre crossvalDataGen --path /home/user/path_to_config --modalities /list/of/modalities --kfolds number/of/folds/to/split --whichos operating_sys
    Args:
        whichos: linux/windows/remote
        path: path to the config file that has data paths
        modalities (list[str]): pass a list of modalities to generate files for crossval subjects
        kfolds: number of crossvalidation folds
        holdout (bool): take out test data for holdout yes if included in CMDline else no

    '''
    folds = int(kfolds)
    
    if isinstance(modalities, str):
        modalities = modalities.strip('""][').replace("'","").split(',')
    modalities = sorted(modalities, reverse=False)
    configs = trainvaltest.return_configs_asdict(whichos, path)
    basepath = os.path.join(configs['os'][whichos]['after_split_path'],'fold1')
    modalityPath = os.path.join(basepath, '_'.join(modalities))
    combine_train_test_val(modalityPath, kfolds=folds)
    return

def combine_train_test_val(path, kfolds=1):
    '''
    For a given modality combines the subjects of train val test folders and stores in a file + generates 5 folds CV data from all subjects and stores 5 files each with train and test subjects
    Args:
        path: path of the modality whose children folders are train val test
    '''
    paths = {'AML':list(), 'CCRCC':list()}
    all_classes_path = glob.glob(path+'/*/*/*')

    for subject in all_classes_path:
        subjectCopy = subject[:]
        clas = subjectCopy.rsplit(os.path.sep, 2)[-2] #AML or CCRCC
        paths[clas].append(subject)

    #working here !!!!!
    with open(os.path.join(path,'allSubjectPaths.yaml'),'w') as file:
        yaml.dump(paths,file)
    k5 = KFold(n_splits=kfolds, shuffle=False)
    train_full = {0:list(),1:list(),2:list(),3:list(),4:list()}
    test_full = {0:list(),1:list(),2:list(),3:list(),4:list()}
    for clas in ['AML', 'CCRCC']:
        fold_num = 0
        for train_index, test_index in k5.split(paths[clas]):
            print("Fold:{}".format(fold_num))
            print(train_index, test_index)
            train_full[fold_num].extend(np.take(paths[clas],train_index))
            test_full[fold_num].extend(np.take(paths[clas],test_index))
            fold_num+=1

    for i in range(5):
        random.shuffle(train_full[i])
        random.shuffle(test_full[i])
        new_train_full = [str(pat) for pat in train_full[i]]
        new_test_full = [str(pat) for pat in test_full[i]]
        store = {'train':new_train_full,
                'test':new_test_full}
        with open(os.path.join(path,'allSubjectPaths{}.yaml'.format(i)),'w') as file:
            yaml.dump(store,file)
    
             