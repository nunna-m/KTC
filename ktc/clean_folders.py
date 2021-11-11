import os
import shutil
import glob
path = '/home/maanvi/LAB/Datasets/kt_new_trainvaltest'

sub_paths = glob.glob(path+'/*/*/*/*')
#print(sub_paths)

for sub in sub_paths:
    for base, dirs, files in os.walk(sub):
        if len(dirs)>0:
            for dir in dirs:
                modals = os.path.join(base,dir)
                need = modals.rsplit('/',5)[1].split('_')
                need_L = [i+'L' for i in need]
                need.extend(need_L)
                if dir not in need:
                    shutil.rmtree(modals)
# for base, dirs, files in os.walk(path):
#     play =  base
#     print(play)
    # modal = play.rsplit('/',5)
    # print(modal[1],modal[-1])
    # req = modal[1].split('_')
    # req_l = [i+'L' for i in req]
    # new_req = req+req_l
    #if modal[-1] not in new_req:
        

