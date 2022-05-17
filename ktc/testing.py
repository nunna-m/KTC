import os
import shutil 
import math
import random
import yaml
import glob
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

modality = 'dc_ec_pc'
num_mods = 3
folds = 10
# datapath = '/home/maanvi/LAB/Datasets/kt_new_trainvaltest/fold1/dc_ec_pc/10CV/allSubjectPaths0.yaml'
# train_subjects, test_subjects = [], []
# with open(datapath,'r') as file:
#     data = yaml.safe_load(file)
# train_subjects.extend(data['train'])
# test_subjects.extend(data['test'])
# mapping = {'AML': 0, 'CCRCC':1}

def read_img(path, file):
    imgpath = os.path.join(path, file)
    labelpath = os.path.join(path+'L', file)
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)

    #crop out exact tumor ROI to the pixel and resize to standard size
    cv2.imwrite('/home/maanvi/Desktop/mask.png',mask)
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    copy = orig_image.copy()
    copy[thresh1==0] = 0
    out = np.zeros_like(copy)
    out[mask == 255] = copy[mask == 255]
    #crop out
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    out = cv2.resize(out, (224,224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('/home/maanvi/Desktop/segmented.png',out)


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


    #gaussian standardizes only modality am
    tmp = imgpath.rsplit(os.path.sep,2)[1]
    if tmp=='am':
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = (orig_image - mean)/std
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = np.clip(orig_image, -1.0, 1.0)
        orig_image = (orig_image + 1.0) / 2.0
        orig_image *= 255
    backup = orig_image[y1:y2,x1:x2]
    backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
    cv2.imwrite('/home/maanvi/Desktop/box.png',backup)
    backup = tf.convert_to_tensor(backup, dtype=tf.uint8)
    return backup


def gen_data_from_paths(subject_paths):
    retX = []
    rety = []
    for path in subject_paths:
        parts = path.rsplit(os.path.sep, 4)
        mods = parts[1].split('_')
        y = mapping[parts[3]]
        filenames = [x[0] for x in os.listdir(os.path.join(path, mods[0]))]#filename just number so only 2 if file is 2.png
        #print("Filenames: ",filenames)
        X = []
        XDict = {i: list() for i in filenames}
        #print(XDict)
        for mod in mods:
            newPath = os.path.join(path, mod)
            for file in os.listdir(newPath):
                num = file[0]
                XDict[num].append(read_img(newPath, file))
        
        for num in XDict:
            temp = np.dstack(XDict[num])
            X.append(temp)
            #print("num: {} X shape: {} y={}".format(num, temp.shape, y))
        X = np.stack(X)
        retX.append(X)
        rety.extend([y]*len(filenames))
    return np.vstack(retX), np.vstack(rety)

#X_train, y_train = gen_data_from_paths(train_subjects)
#X_test, y_test = gen_data_from_paths(test_subjects)
#print(X_train.shape, y_train.shape)

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# print(X_train.shape, y_train.shape)

#plotting data
#dataset = tf.data.Dataset.from_tensor_slices((X_train[0:2]/255.0).astype(np.float32))

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((8 * n_images, 8 * samples_per_image, num_mods))
    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*8:(row+1)*8] = np.vstack(images.numpy())
        row += 1
    
    plt.figure()
    plt.imshow(output)
    plt.show()

#plot_images(dataset, n_images=2, samples_per_image=4)

read_img('/home/maanvi/LAB/Datasets/kt_new_trainvaltest/dc/train/AML/87137931/dc','1.png')