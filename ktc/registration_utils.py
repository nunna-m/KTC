from cProfile import label
from genericpath import isfile
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import shutil

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
    # allPaths = []
    # for modalitycombi in os.listdir(source):
    #     modalities = modalitycombi.split('_')
    #     if len(modalities) == 2:
    #         for split_type in ['train','test','val']:
    #             for clas in ['AML','CCRCC']:
    #                 for subject in os.listdir(source/modalitycombi/split_type/clas):
    #                     subject_path = str(source/modalitycombi/split_type/clas/subject)
    #                     sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
    #                     for sliceName in sameSlicePaths:
    #                         if 'am' in modalities:
    #                             modalities.remove('am')
    #                             modalities = modalities + ['am']
    #                         addThis = (
    #                             os.path.join(subject_path,modalities[0],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[0]+'L',sliceName),
    #                             os.path.join(subject_path,modalities[1],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[1]+'L',sliceName),
    #                             os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
    #                         )
    #                         output = ','.join(addThis)
    #                         output += "\n"
    #                         allPaths.append(output)
    
    # with open('2ModalitiesFilePathsLabels.txt','w') as fp:
    #     for line in allPaths:
    #         fp.write(line)
    

    allPaths = []
    for modalitycombi in os.listdir(source):
        modalities = modalitycombi.split('_')
        if modalities == ['am','dc','ec'] or modalities == ['am','dc','tm']:#if len(modalities) == 3:
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
    
    with open('3ModalitiesFilePathsLabelsReduced.txt','w') as fp:
        for line in allPaths:
            fp.write(line)

    # allPaths = []
    # for modalitycombi in os.listdir(source):
    #     modalities = modalitycombi.split('_')
    #     if len(modalities) == 5:
    #         for split_type in ['train','test','val']:
    #             for clas in ['AML','CCRCC']:
    #                 for subject in os.listdir(source/modalitycombi/split_type/clas):
    #                     subject_path = str(source/modalitycombi/split_type/clas/subject)
    #                     sameSlicePaths = getIntersectionFileNames(subject_path=subject_path,modalities=modalities)
    #                     for sliceName in sameSlicePaths:
    #                         if 'am' in modalities:
    #                             modalities.remove('am')
    #                             modalities = modalities + ['am']
    #                         addThis = (
    #                             os.path.join(subject_path,modalities[0],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[0]+'L',sliceName),
    #                             os.path.join(subject_path,modalities[1],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[1]+'L',sliceName),
    #                             os.path.join(subject_path,modalities[2],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[2]+'L',sliceName),
    #                             os.path.join(subject_path,modalities[3],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[3]+'L',sliceName),
    #                             os.path.join(subject_path,modalities[4],sliceName),
    #                             os.path.join(subject_path.replace('kt_new_trainvaltest','kt_labels'),modalities[4]+'L',sliceName),
    #                             os.path.join(dest/modalitycombi/split_type/clas/subject,sliceName)
    #                         )
    #                         output = ','.join(addThis)
    #                         output += "\n"
    #                         allPaths.append(output)
    
    # with open('5ModalitiesFilePathsLabels.txt','w') as fp:
    #     for line in allPaths:
    #         fp.write(line)

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
        if modalities == ['am','dc','ec'] or modalities == ['am','dc','tm']:#if len(modalities) == 3:
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
    
    with open('3ModalitiesFilePathsReduced.txt','w') as fp:
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
    imagePath = '/home/maanvi/LAB/Datasets/kt_registered/am_dc/train/CCRCC/17616386/1.png'
    image = cv2.imread(imagePath)
    print(image.shape)
    channel1 = image[:,:,0]
    channel2 = image[:,:,1]
    channel3 = image[:,:,2]
    cv2.imwrite('/home/maanvi/channel1.png',channel1)
    cv2.imwrite('/home/maanvi/channel2.png',channel2)
    cv2.imwrite('/home/maanvi/channel3.png',channel3)

def removeSubjects():
    source = Path('/home/maanvi/LAB/Datasets/kt_registered')
    removesubjectsFilePath = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/remove_subjects.txt'
    with open(removesubjectsFilePath,'r') as fp:
        for subject in fp:
            subject = subject.rstrip()
            subject_path = source/subject
            if os.path.isfile(subject_path):
                os.remove(subject_path)
                print("removing file")
                print(subject_path)
            elif os.path.isdir(subject_path):
                shutil.rmtree(str(subject_path))
                print(subject_path)
            

def writeRegTumorExact(imgpath,labelpath,destpath):
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = orig_image.shape
    print(imgpath)
    mask = cv2.imread(labelpath)[:,:,0]
    mask = cv2.resize(mask, (orig_width,orig_height),cv2.INTER_CUBIC)
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = (orig_image - mean)/std
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = np.clip(orig_image, -1.0, 1.0)
    orig_image = (orig_image + 1.0) / 2.0
    orig_image *= 255
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    orig_image[thresh1==0] = 0
    out = np.zeros_like(orig_image)
    out[mask == 255] = orig_image[mask == 255]
    #crop out
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    out = cv2.resize(out, (224,224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(destpath,out)

def writeRegTumorBox(imgpath,labelpath,destpath):
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    mask = cv2.imread(labelpath)[:,:,0]
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    assert (w,h)<(224,224)
    assert (x,y)>=(0,0)
    const = 0.3
    diff_x = int(const*w)
    diff_y = int(const*h)
    if (x-diff_x)<0:
        x1 = 0
    else:
        x1 = x-diff_x
    if (y-diff_y)<0:
        y1 = 0
    else:
        y1 = y-diff_y
    if (x+w+diff_x)>=orig_width:
        x2 = orig_width
    else:
        x2 = x+diff_x+w
    if (y+diff_y+h)>=orig_height:
        y2 = orig_height
    else:
        y2 = y+diff_y+h
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = (orig_image - mean)/std
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = np.clip(orig_image, -1.0, 1.0)
    orig_image = (orig_image + 1.0) / 2.0
    orig_image *= 255
    backup = orig_image[y1:y2,x1:x2]
    backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(destpath,backup)

def storeRegisteredCroppedImg(cropType='exact'):
    #source: /home/maanvi/LAB/Datasets/kt_registered, /home/maanvi/LAB/Datasets/kt_registered_labels
    #dest: /home/maanvi/LAB/Datasets/kt_registered_box and /home/maanvi/LAB/Datasets/kt_registered_exact
    dest = '/home/maanvi/LAB/Datasets/kt_registered_box'
    img_source = Path('/home/maanvi/LAB/Datasets/kt_registered')
    label_source = Path(str(img_source).replace('kt_registered','kt_registered_labels'))
    for modalitycombi in os.listdir(img_source):
        modalities = modalitycombi.split('_')
        for split_type in ['train','test','val']:
            for clas in ['AML','CCRCC']:
                for subject in os.listdir(img_source/modalitycombi/split_type/clas):
                    subject_path = str(img_source/modalitycombi/split_type/clas/subject)
                    for img in os.listdir(img_source/modalitycombi/split_type/clas/subject):
                        imgpath = str(img_source/modalitycombi/split_type/clas/subject/img)
                        labelpath = str(label_source/modalitycombi/split_type/clas/subject/img)
                        if cropType == 'box':
                            destpath = labelpath.replace('kt_registered_labels','kt_registered_box')
                            dest_folder_path = destpath[:-6]
                            os.makedirs(dest_folder_path,exist_ok=True)
                            writeRegTumorBox(imgpath,labelpath,destpath)
                        else:
                            destpath = labelpath.replace('kt_registered_labels','kt_registered_exact')
                            dest_folder_path = destpath[:-6]
                            os.makedirs(dest_folder_path,exist_ok=True)
                            writeRegTumorExact(imgpath,labelpath,destpath)
                        
                        print(destpath)



if __name__ == "__main__":
    #source = Path('/home/maanvi/LAB/Datasets/kt_new_trainvaltest')
    #dest = Path('/home/maanvi/LAB/Datasets/kt_registered_labels')
    #copyFolderStructure(source, dest)
    #createPathFiles2ModalitiesJSON(source, dest)
    #createPathFiles3ModalitiesJSON(source, dest)
    #createPathFiles5ModalitiesJSON(source, dest)
    #filterRequiredPaths()
    #change_subject_path()
    #displayRegisteredImage()
    #storeImageLabelMasks()
    #storeImageLabelMask_FilePaths()
    #removeSubjects()
    storeRegisteredCroppedImg(cropType='box')
    storeRegisteredCroppedImg(cropType='exact')

