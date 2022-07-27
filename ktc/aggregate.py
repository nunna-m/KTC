import os
import newCrossVal
import newFolders

#datasets_root = '/home/maanvi/LAB/Datasets'
datasets_root = '/kw_resources/datasets'
oldPath = os.path.join(datasets_root,'kt_new_trainvaltest')
newPath = os.path.join(datasets_root,'kt_combined')
os.makedirs(newPath, exist_ok=True)


def main():
    #first create raw data folder in ktcombined
    newFolders.createRawDataFolder(oldPath=oldPath, newPath=newPath)
    for modFolder in os.listdir(oldPath):
        path = os.path.join(newPath,modFolder,'rawData')
        newCrossVal.createFolds(basePath=path)
    newFolders.createNumpyFiles(oldPath=oldPath, newPath=newPath)


main()
    