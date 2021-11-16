import os

def old_count_samples(modalities, generalpath, split):
    modalities = sorted(modalities, reverse=False)
    ms = '_'.join(modalities)
    aml = os.path.join(generalpath,ms,split,'AML')
    nf_aml = 0
    for base, dirs, files in os.walk(aml):
        get_mode = base.rsplit('/',1)[1]
        if get_mode in modalities:
            for file in os.walk(base):
                print("filenames: ",file[0],file[2])
                nf_aml+=len(file[2])

    nf_cc = 0
    cc = os.path.join(generalpath,ms,split,'CCRCC')
    for base, dirs, files in os.walk(cc):
        get_mode = base.rsplit('/',1)[1]
        if get_mode in modalities:
            for file in os.walk(base):
                nf_cc+=len(file[2])

    return {
        'AML':nf_aml,
        'CCRCC':nf_cc,
        'total':nf_aml+nf_cc,
    }

#print(count_samples(['ec','tm'],'/home/maanvi/LAB/Datasets/kt_new_trainvaltest', 'val'))


def count_samples(modalities, generalpath, split):
    modalities = sorted(modalities, reverse=False)
    ms = '_'.join(modalities)
    aml = os.path.join(generalpath,ms,split,'AML')
    nf_aml = 0
    for subject in os.listdir(aml):
        subject_path = os.path.join(aml, subject)
        count = 10
        for mode in os.listdir(subject_path):
            if mode in modalities:
                mode_path = os.path.join(aml, subject, mode)
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

print(count_samples(['ec','tm'],'/home/maanvi/LAB/Datasets/kt_new_trainvaltest', 'val'))