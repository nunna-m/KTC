import os
import shutil 
import math
import random
import yaml
import glob
from sklearn.model_selection import KFold
import numpy as np

path = r'D:\\01_Maanvi\\LABB\\datasets\\kt_new_trainvaltest\\fold1\\am_dc_ec_pc_tm'
paths = {'AML':list(), 'CCRCC':list()}
all_classes_path = glob.glob(path+'/*/*/*')

for subject in all_classes_path:
    subjectCopy = subject[:]
    clas = subjectCopy.rsplit(os.path.sep, 2)[-2]
    paths[clas].append(subject)

with open(os.path.join(path,'allSubjectPaths.yaml'),'w') as file:
    yaml.dump(paths,file)

k5 = KFold(n_splits=5, shuffle=False)
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