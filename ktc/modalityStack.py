import numpy as np
import os
import cv2

def get_tumor_boundingbox(imgpath, labelpath):
    '''
    get the bounding box coordinates around tumor
    first calculate center of tumor based on segmentation label
    then calculate bounding box around it after zooming out by a factor of 0.3 on both heigth and width (just to be sure of including the entire region of the tumor)
    am modality is gaussian standardized also
    '''
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)
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


    #gaussian standardize all modalities
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = (orig_image - mean)/std
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = np.clip(orig_image, -1.0, 1.0)
    orig_image = (orig_image + 1.0) / 2.0
    orig_image *= 255
    backup = orig_image[y1:y2,x1:x2]
    backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_CUBIC)
    mod = imgpath.rsplit(os.path.sep,2)[1]
    #cv2.imwrite(f'/home/maanvi/Desktop/boxCrop{mod}.png',backup)
    return backup

def get_exact_tumor(imgpath, labelpath):
    '''
    get the exact segmented tumor region (pixel perfect) based on label already provided
    '''
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    #gaussian standardizes all modalities
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = (orig_image - mean)/std
    mean, std = orig_image.mean(), orig_image.std()
    orig_image = np.clip(orig_image, -1.0, 1.0)
    orig_image = (orig_image + 1.0) / 2.0
    orig_image *= 255
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)
    #cv2.imwrite('/home/maanvi/Desktop/mask.png',mask)
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
    mod = imgpath.rsplit(os.path.sep,2)[1]
    #cv2.imwrite(f'/home/maanvi/Desktop/pixelCrop{mod}.png',out)
    return out

def get_orig_image(imgpath, labelpath):
    image = cv2.imread(imgpath)[:,:,0]
    image = cv2.resize(image, (224,224),interpolation = cv2.INTER_CUBIC)
    mod = imgpath.rsplit(os.path.sep,2)[1]
    #cv2.imwrite(f'/home/maanvi/Desktop/actual{mod}.png',image)
    return image

def getImage(imagePath, labelPath, cropType=None):
    if cropType is None:
        #read image full and pass back
        return get_orig_image(imagePath,labelPath)
    
    if cropType == 'center':
        #center crop and pass back
        return get_tumor_boundingbox(imagePath, labelPath)
    
    if cropType == 'pixel':
        #crop based on segmentation label and pass back
        return get_exact_tumor(imagePath, labelPath)


def getSubjectData(subject_path, cropType):
    #print(cropType)
    pathParts = subject_path.rsplit(os.path.sep, 4)
    modalities = pathParts[1].split('_')
    subject_data = {}
    subject_data['path'] = subject_path
    subject_data['modalities'] = modalities
    subject_data['clas'] = str(pathParts[3])
    subject_data['ID'] = str(pathParts[4])
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
    #crop type mapping: {'centerCrop'->'center','pixelCrop'->'pixel',None->'fullImage'}
    cropTypeMapping = {
        'centerCrop':'center',
        'pixelCrop':'pixel',
        'fullImage': None
    }
    for modality,names in gathered_modalityPaths.items():
        subject_data[modality] = {
            os.path.splitext(name)[0]:getImage(os.path.join(subject_path,modality,name),os.path.join(subject_path,modality+'L',name),cropTypeMapping[cropType]) for name in names
        }

    #print(subject_data)
    return subject_data

def combineData(subjectPath, storePath, cropType):
    # tempSubjectPath = '/home/maanvi/LAB/Datasets/kt_new_trainvaltest/am_pc_tm/train/CCRCC/17672172'
    subject_data = getSubjectData(subjectPath, cropType)
    #print(subject_data['path'])
    modalities = subject_data['path'].rsplit(os.path.sep,4)[1]
    try:
        if len(modalities) > 2:
            assert '_' in modalities
            modalities = modalities.split('_')
        else:
            modalities = [f'{modalities}']
    except AssertionError:
        print(f"need modailities with _ but got this instead '{modalities}'")

    modalities = sorted(modalities, reverse=False)
    #print(modalities)
    imageNames = list(subject_data[modalities[0]].keys())
    for name in imageNames:
        image = None
        for modality in modalities:
            if image is None:
                image = np.array(subject_data[modality][name])
            else:
                image = np.dstack([image, np.array(subject_data[modality][name])])
        
        
        if len(modalities) == 1:
            #one modality, duplicate thrice
            image = np.repeat(image[:,:,np.newaxis],3,axis=-1)
        elif len(modalities) == 2:
            #two modalities, add 0 array as third dimension
            zeros = np.zeros((image.shape[0],image.shape[1],1))
            # print(f'{zeros.shape} zeros shape')
            # print(f'{image.shape} prev img shape')
            image = np.dstack((image,zeros))
        
        #print(f"{name}.png -- shape {image.shape}")
        filename = f"{subject_data['ID']}_{subject_data['clas']}_{name}.npz"
        labelMapping = {'AML':0,'CCRCC':1}
        #data = {'image':image,'label':labelMapping[subject_data['clas']]}
        np.savez(os.path.join(storePath,filename),image=image,label=labelMapping[subject_data['clas']])
        #cv2.imwrite(f'/home/maanvi/Desktop/combined{name}.png',image)
    
    return image


#next tasks
#1. store three folders fullimages, centercrop, pixelCrop
#2. borrow getTumorBoundingBox and getExactTumor functions from dataset.py
#3. get the images and store in numpy array format and store the images as subjectID_AML_1.npy, subjectID_AML_2.npy... in subject folder store image npy along with label? or write class in filename
#4. get the images and store in RGB image format and store the images as subjectID_1.png, subjectID_2.png... in subject folder for all modalities <= 3 (duplicate thrice, append 0s or use all 3)
#5. another file write function to go to every subject and collect images, print out shape and check if shape is matching
#6. store training and testing collected numpy arrays .npys for every fold

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
'''