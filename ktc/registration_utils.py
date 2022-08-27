import os
import sys
from pathlib import Path

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

def filterRequiredPaths():
    #filename: /home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePaths.txt
    filepath = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePaths.txt'
    newfilepath = '/home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePathsReduced.txt'
    with open(filepath,'r') as read_here, open(newfilepath,'w') as write_here:
        for line in read_here:
            if 'am_tm' in line or 'dc_ec' in line or 'dc_pc' in line or 'ec_pc' in line:
                write_here.write(line)


if __name__ == "__main__":
    source = Path('/home/maanvi/LAB/Datasets/kt_new_trainvaltest')
    dest = Path('/home/maanvi/LAB/Datasets/kt_registered')
    #copyFolderStructure(source, dest)
    createPathFiles2ModalitiesJSON(source, dest)
    filterRequiredPaths()

