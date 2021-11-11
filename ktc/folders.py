import os
aml = '/home/maanvi/LAB/Datasets/kt_new_trainvaltest/dc_ec/test/AML'
typ = aml.rsplit('/',2)[-2]
gen_path = aml.rsplit('/',1)[0]
cc = gen_path+'/CCRCC'

nf_aml = 0
nf_cc = 0

modals = aml.rsplit('/',4)[-3].split('_')

for base, dirs, files in os.walk(aml):
    get_mode = base.rsplit('/',1)[1]
    if get_mode in modals:
        for file in os.walk(base):
            nf_aml+=len(file[2])

for base, dirs, files in os.walk(cc):
    get_mode = base.rsplit('/',1)[1]
    if get_mode in modals:
        for file in os.walk(base):
            nf_cc+=len(file[2])

print('type: {}'.format(typ))
print("{}--AML files: {} ".format(' '.join(modals),nf_aml))
print("{}--CCRCC files: {}".format(' '.join(modals),nf_cc))