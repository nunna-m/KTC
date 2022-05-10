import os
import yaml

def count_samples(modalities, generalpath, split):
    '''
    count number of samples for each modality and tumor class based on split
    returns a dictionary of number of images in each tumor class and the sum of all the numbers
    Args:
        modalities: the modalities that are being considered for training
        generalpath: subject directory that has train/val/test folders
        split: either train or val or test split
    '''
    modalities = sorted(modalities, reverse=False)
    ms = '_'.join(modalities)
    aml = os.path.join(generalpath,ms,split,'AML')
    print(modalities, generalpath, ms, aml)
    nf_aml = 0
    for subject in os.listdir(aml):
        subject_path = os.path.join(aml, subject)
        count = 10
        for mode in os.listdir(subject_path):
            if mode in modalities:
                mode_path = os.path.join(subject_path, mode)
                listout = os.listdir(mode_path)
                if len(listout) < count:
                    count = len(listout)
        nf_aml += count

    cc = os.path.join(generalpath,ms,split,'CCRCC')
    nf_cc = 0
    for subject in os.listdir(cc):
        subject_path = os.path.join(cc, subject)
        count = 10
        for mode in os.listdir(subject_path):
            if mode in modalities:
                mode_path = os.path.join(cc, subject, mode)
                listout = os.listdir(mode_path)
                if len(listout) < count:
                    count = len(listout)
        nf_cc += count
    
    return {
        'AML':nf_aml,
        'CCRCC':nf_cc,
        'total':nf_aml+nf_cc,
    }

def count_fromFiles(path, split):
    '''
    takes a crossvalidation fold file path, and returns count of images within that folder
    returns two values of number of images in AML class and number of images in CCRCC class
    Args:
        path: CV fold path consists of training subject paths and testing subject paths
        split: train or test, based on which count is performed by doing listdir and count for each class is returned
    '''
    with open(path, 'r') as file:
        data = yaml.safe_load(file)[split]
    
    classes = {'AML':0,'CCRCC':0}
    for subject in data:
        parts = subject.rsplit(os.path.sep, 4)
        mods = parts[1].split('_')
        clas = parts[3]
        #print(clas)
        classes[clas] += len(os.listdir(os.path.join(subject,mods[0])))
    
    return classes['AML'], classes['CCRCC']

def count_total(path, split):
    '''
    return total number of images in both AML and CCRCC combined
    '''
    aml, cc = count_fromFiles(path, split)
    return aml+cc



# print(count_samples(['dc'],'/home/maanvi/LAB/Datasets/kt_new_trainvaltest/fold1', 'train'))