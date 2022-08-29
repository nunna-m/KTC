import os
import sys
from pathlib import Path
import cv2
import numpy as np

def copyFolderStructure(source, dest):
    #source: /home/maanvi/LAB/Datasets/kt_new_trainvaltest/
    #   dc_pc/train/AML/16639185/dc
    #dest: /home/maanvi/LAB/Datasets/kt_registered/
    #   dc_pc/train/AML/16639185/
    for modalitycombi in os.listdir(source):
        for split_type in ['train','test','val']:
            for clas in ['AML','CCRCC']:
                for subject in os.listdir(source/modalitycombi/split_type/clas):
                    os.makedirs(dest/modalitycombi/split_type/clas/subject,exist_ok=True)


def getIntersectionFileNames(subject_path, modalities):
    gathered_modalities_paths = {
            modality: set(os.listdir(os.path.join(subject_path,modality)))
            for modality in modalities
        }
    same_named_slices = set.intersection(*map(
        lambda slices: set(
            map(lambda name: os.path.splitext(name)[0], slices)),
        gathered_modalities_paths.values(),
        ))
    for temp in modalities:
        gathered_modalities_paths[temp] = {k+'.png' for k in same_named_slices}

    for modality in modalities:
        gathered_modalities_paths[modality] = list(
            filter(lambda x: os.path.splitext(x)[0],
            gathered_modalities_paths[modality])
        )
    return gathered_modalities_paths[modalities[0]]

def convertToMask(imgpath):
    orig_image = cv2.imread(imgpath)[:,:,0]
    #cv2.imwrite(f'/home/maanvi/registered_{imgpath[-5]}.png',orig_image)
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    imgpath_parts = imgpath.rsplit(os.path.sep,2)
    imgpath_parts[1] += 'L'
    labelpath = '/'.join(imgpath_parts)
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)
    new_labelpath = labelpath.replace('kt_new_trainvaltest','kt_labels')
    direc = new_labelpath[:-5]
    os.makedirs(direc, exist_ok=True)
    #print(direc)
    cv2.imwrite(new_labelpath,mask)

def storeImageLabelMasks():
    modalities_file_path = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/5ModalitiesFilePaths.txt'
    with open(modalities_file_path,'r') as fp:
        for line in fp:
            parts = line.rstrip().split(',')[:-1]
            for part in parts:
                convertToMask(part)

def storeImageLabelMask_FilePaths():
    source = Path('/home/maanvi/LAB/Datasets/kt_new_trainvaltest/')
    dest = Path('/home/maanvi/LAB/Datasets/kt_registered_labels/')
    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 2:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            if 'am' in modalities:
                                modalities.remove('am')
                                modalities = modalities + ['am']
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[0]+'L',sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[1]+'L',sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('2ModalitiesFilePathsLabels.txt','w') as fp:
        for line in allPaths:
            fp.write(line)
    

    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 3:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            if 'am' in modalities:
                                modalities.remove('am')
                                modalities = modalities + ['am']
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[0]+'L',sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[1]+'L',sliceName),
                                os.path.join(subject_path,modalities[2],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[2]+'L',sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('3ModalitiesFilePathsLabels.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 5:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            if 'am' in modalities:
                                modalities.remove('am')
                                modalities = modalities + ['am']
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[0]+'L',sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[1]+'L',sliceName),
                                os.path.join(subject_path,modalities[2],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[2]+'L',sliceName),
                                os.path.join(subject_path,modalities[3],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[3]+'L',sliceName),
                                os.path.join(subject_path,modalities[4],sliceName),
                                os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[4]+'L',sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('5ModalitiesFilePathsLabels.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

def createPathFiles2ModalitiesJSON(source, dest):
    #source: /home/maanvi/LAB/Datasets/kt_new_trainvaltest/
    #dest: /home/maanvi/LAB/Datasets/kt_registered/
    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 2:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('2ModalitiesFilePaths.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

def createPathFiles3ModalitiesJSON(source, dest):
    #source: /home/maanvi/LAB/Datasets/kt_new_trainvaltest/
    #dest: /home/maanvi/LAB/Datasets/kt_registered/
    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 3:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            if 'am' in modalities:
                                modalities.remove('am')
                                modalities = modalities + ['am']
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(subject_path,modalities[2],sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('3ModalitiesFilePaths.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

def createPathFiles5ModalitiesJSON(source, dest):
    #source: /home/maanvi/LAB/Datasets/kt_new_trainvaltest/
    #dest: /home/maanvi/LAB/Datasets/kt_registered/
    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if len(modalities) == 5:
            for split_type in ['train','test','val']:
                for clas in ['AML','CCRCC']:
                    for subject in os.listdir(source/modalitycombi/split_type/clas):
                        subject_path = str(source/modalitycombi/split_type/clas/subject)
                        sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
                        for sliceName in sameSlicePaths:
                            if 'am' in modalities:
                                modalities.remove('am')
                                modalities = modalities + ['am']
                            addThis = (
                                os.path.join(subject_path,modalities[0],sliceName),
                                os.path.join(subject_path,modalities[1],sliceName),
                                os.path.join(subject_path,modalities[2],sliceName),
                                os.path.join(subject_path,modalities[3],sliceName),
                                os.path.join(subject_path,modalities[4],sliceName),
                                os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
                            )
                            output = ','.join(addThis)
                            output += "\n"
                            allPaths.append(output)
    
    with open('5ModalitiesFilePaths.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

def filterRequiredPaths():
    #filename: /home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePaths.txt
    filepath = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePaths.txt'
    newfilepath = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePathsReduced.txt'
    with open(filepath,'r') as read_here, open(newfilepath,'w') as write_here:
        for line in read_here:
            if 'am_tm' in line or 'am_ec' in line or 'am_pc' in line or 'am_dc' in line:
                write_here.write(line)

def change_subject_path():
    oldPath = '/home/maanvi/LAB/Datasets/kt_new_trainvaltest/ec_tm/train/AML/16639185'
    #newPath = '/home/maanvi/LAB/Datasets/kt_registered/ec_tm/train/AML/16639185'
    newPath = oldPath.replace('kt_new_trainvaltest','kt_registered')
    print(f'oldpath: {oldPath}')
    print(f'newpath: {newPath}')

def displayRegisteredImage():
    imagePath = '/home/maanvi/LAB/Datasets/kt_registered/ec_tm/train/AML/16639185/1.png'
    image = cv2.imread(imagePath)
    print(image.shape)
    channel1 = image[:,:,0]
    channel2 = image[:,:,1]
    channel3 = image[:,:,2]
    cv2.imwrite('/home/maanvi/channel1.png',channel1)
    cv2.imwrite('/home/maanvi/channel2.png',channel2)
    cv2.imwrite('/home/maanvi/channel3.png',channel3)

if __name__ == "__main__":
    source = Path('/home/maanvi/LAB/Datasets/kt_new_trainvaltest')
    dest = Path('/home/maanvi/LAB/Datasets/kt_registered_labels')
    #copyFolderStructure(source, dest)
    #createPathFiles2ModalitiesJSON(source, dest)
    #createPathFiles3ModalitiesJSON(source, dest)
    #createPathFiles5ModalitiesJSON(source, dest)
    #filterRequiredPaths()
    #change_subject_path()
    #displayRegisteredImage()
    #storeImageLabelMasks()
    storeImageLabelMask_FilePaths()

