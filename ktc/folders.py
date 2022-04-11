import os
import yaml
def count_samples(modalities, generalpath, split):
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

# print(count_samples(['dc'],'/home/maanvi/LAB/Datasets/kt_new_trainvaltest/fold1', 'train'))

def count_fromFiles(path, split):
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
    aml, cc = count_fromFiles(path, split)
    return aml+cc