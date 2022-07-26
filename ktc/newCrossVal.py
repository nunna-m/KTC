import os
import random
import yaml
import glob
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
import numpy as np

def createFolds(basePath, kfolds, loocv=False):
    #base path is the path of modality folder + rawData whose children are AML and CCRCC
    modalities = basePath.rsplit(os.path.sep,2)[1].split('_')
    paths = {'AML':list(), 'CCRCC':list()}
    all_classes_path = glob.glob(basePath+'/*/*')
    target_classes = []
    for subject in all_classes_path:
        subjectCopy = subject[:]
        print(subjectCopy)


createFolds('/home/maanvi/LAB/Datasets/kt_combined/pc/rawData', 5)