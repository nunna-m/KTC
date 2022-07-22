import numpy as np
import os
'''
subject_data = {
    'subject_path': 'D:\\01_Maanvi\\LABB\\datasets\\kt_new_trainvaltest\\fold1\\am_ec\\train\\AML\\16313384', 'clas': 'AML', \
    'ID': '16313384', 
    'num_slices_per_modality': 2, 
    'am': {
        '2':[[0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 3, 0],
       [0, 0, 2, 0, 0, 5]], 
       '1': [[3, 0, 2, 0, 1, 0],
       [5, 0, 4, 0, 0, 3],
       [5, 0, 0, 0, 1, 1]]
    }, 
    'ec': {
        '2': [[2, 2, 0, 0, 0, 1],
       [1, 1, 0, 0, 5, 0],
       [0, 0, 0, 0, 3, 4]], 
       '1':[[5, 1, 3, 0, 0, 0],
       [0, 4, 0, 2, 0, 0],
       [0, 0, 3, 0, 0, 5],]
    }
}

modalities = subject_data['subject_path'].rsplit(os.path.sep,5)[-4]
try:
    assert '_' in modalities
    modalities = modalities.split('_')
except AssertionError:
    print(f"need modailities with _ but got this instead '{modalities}'")

imageNames = list(subject_data[modalities[0]].keys())
for name in imageNames:
    image = None
    for modality in modalities:
        if image is None:
            image = np.array(subject_data[modality][name])
        else:
            image = np.dstack([image, np.array(subject_data[modality][name])])
    
    print(f"{name}.png -- shape {image.shape}")
'''

#subject_path = r'D:\Ddesktop\16639185'
subject_path = '/home/maanvi/Desktop/74298266'
modalities = ['am','dc','ec','pc','tm']
clas = 'AML'
gathered_modalityPaths = {
    modality: set(
        os.listdir(
            os.path.join(
                subject_path,modality
            )
        )
    )
    for modality in modalities
}
print(gathered_modalityPaths)
same_named_imageNames = set.intersection(
    *map(
        lambda slices:
        set(
            map(
                lambda name: os.path.splitext(name)[0], slices
            )
        ), gathered_modalityPaths.values()
    )
)

for temp in modalities:
        gathered_modalityPaths[temp] = {k+'.png' for k in same_named_imageNames}

for modality in modalities:
        gathered_modalityPaths[modality] = list(
            filter(lambda x: os.path.splitext(x)[0],
            gathered_modalityPaths[modality])
        )
print(gathered_modalityPaths)

imageNames = list(gathered_modalityPaths[modalities[0]].keys())
for name in imageNames:
    image = None
    for modality in modalities:
        if image is None:
            image = np.array(gathered_modalityPaths[modality][name])
        else:
            image = np.dstack([image, np.array(gathered_modalityPaths[modality][name])])
    
    print(f"{name}.png -- shape {image.shape}")
#next tasks
#1. store three folders fullimages, centercrop, pixelCrop
#2. borrow getTumorBoundingBox and getExactTumor functions from dataset.py
#3. get the images and store in numpy array format and store the images as subjectID_AML_1.npy, subjectID_AML_2.npy... in subject folder store image npy along with label? or write class in filename
#4. get the images and store in RGB image format and store the images as subjectID_1.png, subjectID_2.png... in subject folder for all modalities <= 3 (duplicate thrice, append 0s or use all 3)
#5. another file write function to go to every subject and collect images, print out shape and check if shape is matching
#6. store training and testing collected numpy arrays .npys for every fold
