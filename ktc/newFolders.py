import os
import yaml
import shutil
import modalityStack


def createNPZFiles(basePath,trainSubjects,testSubjects,foldType, foldNum):
    rawDataPath = os.path.join(basePath,'rawData')
    newDataPath = os.path.join(basePath,'numpyData')

    for filename in trainSubjects:
        subjectID, clas = filename.split('_')
        source = os.path.join(rawDataPath,clas,subjectID)
        for typ in ['fullImage','centerCrop','pixelCrop']:
            dest = os.path.join(newDataPath,typ,'train',foldType,foldNum)
            os.makedirs(dest,exist_ok=True)
            arr = modalityStack.combineData(source,dest,typ)
            #print(f'checking image shape: {arr.shape}')
    
    for filename in testSubjects:
        subjectID, clas = filename.split('_')
        source = os.path.join(rawDataPath,clas,subjectID)
        for typ in ['fullImage','centerCrop','pixelCrop']:
            dest = os.path.join(newDataPath,typ,'test',foldType,foldNum)
            os.makedirs(dest,exist_ok=True)
            arr = modalityStack.combineData(source,dest,typ)
            #print(f'checking image shape: {arr.shape}')


def createRawDataFolder(oldPath, newPath):
    trainvaltest = ['train','val','test']
    classes = ['AML','CCRCC']
    for modFolder in os.listdir(oldPath):
        os.makedirs(os.path.join(newPath,modFolder),exist_ok=True)
        os.makedirs(os.path.join(newPath,modFolder,'rawData'),exist_ok=True)
        for clas in classes:
            os.makedirs(os.path.join(newPath,modFolder,'rawData',clas),exist_ok=True)
            for subfold in ['train','test']:
                os.makedirs(os.path.join(newPath,modFolder,'numpyData','fullImage',subfold),exist_ok=True)
                os.makedirs(os.path.join(newPath,modFolder,'numpyData','centerCrop',subfold),exist_ok=True)
                os.makedirs(os.path.join(newPath,modFolder,'numpyData','pixelCrop',subfold),exist_ok=True)

    for modFolder in os.listdir(oldPath):
        for splitType in trainvaltest:
            for clas in classes:
                source = os.path.join(oldPath,modFolder,splitType,clas)
                dest = os.path.join(newPath,modFolder,'rawData',clas)
                for subjectID in os.listdir(source):
                    shutil.copytree(os.path.join(source,subjectID),os.path.join(dest,subjectID),dirs_exist_ok=True)

    # for modFolder in os.listdir(oldPath):
    #     for clas in classes:
    #         currentPath = os.path.join(newPath,modFolder,'rawData',clas)
    #         for subjectID in os.listdir(currentPath):
    #             source = os.path.join(currentPath,subjectID)
    #             for typ in ['fullImage','centerCrop','pixelCrop']:
    #                 dest = os.path.join(newPath,modFolder,'numpyData',typ)
    #                 arr = modalityStack.combineData(source,dest,typ)
    #                 print(f'checking image shape: {arr.shape}')

def createNumpyFiles(oldPath, newPath):
    for modFolder in os.listdir(oldPath):
        for folder in ['5CV','10CV','LOOCV']:
            foldsPath = os.path.join(newPath,modFolder,'foldDataFiles',folder)
            for foldD in os.listdir(foldsPath):
                foldNum = os.path.splitext(foldD)[0][-1]
                with open(os.path.join(foldsPath,foldD),'r') as file:
                    data = yaml.safe_load(file)
                createNPZFiles(os.path.join(newPath,modFolder),data['train'],data['test'],folder, foldNum)
