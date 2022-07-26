from math import modf
import os
import yaml
import shutil
import modalityStack

# def saveNumpy(source,destination):
#     modalityStack.combineData(source,destination)
datasets_root = '/home/maanvi/LAB/Datasets'
oldPath = os.path.join(datasets_root,'sample_kt')
newPath = os.path.join(datasets_root,'kt_combined')
os.makedirs(newPath, exist_ok=True)

trainvaltest = ['train','val','test']
classes = ['AML','CCRCC']
for modFolder in os.listdir(oldPath):
    os.makedirs(os.path.join(newPath,modFolder),exist_ok=True)
    os.makedirs(os.path.join(newPath,modFolder,'rawData'),exist_ok=True)
    for clas in classes:
        os.makedirs(os.path.join(newPath,modFolder,'rawData',clas),exist_ok=True)
        os.makedirs(os.path.join(newPath,modFolder,'numpyData','fullImage',clas),exist_ok=True)
        os.makedirs(os.path.join(newPath,modFolder,'numpyData','centerCrop',clas),exist_ok=True)
        os.makedirs(os.path.join(newPath,modFolder,'numpyData','pixelCrop',clas),exist_ok=True)

for modFolder in os.listdir(oldPath):
    for splitType in trainvaltest:
        for clas in classes:
            source = os.path.join(oldPath,modFolder,splitType,clas)
            dest = os.path.join(newPath,modFolder,'rawData',clas)
            for subjectID in os.listdir(source):
                shutil.copytree(os.path.join(source,subjectID),os.path.join(dest,subjectID),dirs_exist_ok=True)

for modFolder in os.listdir(oldPath):
    for clas in classes:
        currentPath = os.path.join(newPath,modFolder,'rawData',clas)
        for subjectID in os.listdir(currentPath):
            source = os.path.join(currentPath,subjectID)
            for typ in ['fullImage','centerCrop','pixelCrop']:
                dest = os.path.join(newPath,modFolder,'numpyData',typ,clas)
                arr = modalityStack.combineData(source,dest,typ)
                print(f'checking image shape: {arr.shape}')
