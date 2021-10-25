import os
from numpy.lib import index_tricks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import cv2

#paths
path = '/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/CCRCC/16269495/am{}/1.png'
imgpath = path.format('')
labelpath = path.format('L')
storepath = '/home/maanvi/Desktop/working/'

# #read images
# index=0
# init = cv2.imread(imgpath)
# image = cv2.imread(imgpath)[:,:,index]
# image_label = cv2.resize(cv2.imread(labelpath), (image.shape[1],image.shape[0]))[:,:,index]
# cv2.imwrite(storepath+'image_{}.png'.format(index),image)


# thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# x,y,w,h = cv2.boundingRect(thresh)
# otsu = image_label[y:y+h,x:x+w]
# cv2.imwrite(storepath+'otsu_image.png',otsu)


# lower = np.array([0], np.uint8)
# upper = np.array([0], np.uint8)
# mask = cv2.inRange(otsu, lower, upper)
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
# c = max(contours, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(c)
# rect = cv2.rectangle(init, (x,y), (x+w,y+h), (0,0,255),2)
# final =image[y:y+h,x:x+w]
# cv2.imwrite(storepath+'cropped_image.png',rect)

# orig_img = cv2.imread(imgpath)[:,:,0]
# (orig_height, orig_width) = orig_img.shape
# image = cv2.imread(labelpath)
# image = cv2.resize(image, (orig_width, orig_height))
# backup = image.copy()
# lower_red = np.array([0,0,50])
# upper_red = np.array([0,0,255])
# mask = cv2.inRange(image, lower_red, upper_red)
# cv2.imwrite(storepath+'mask.png',mask)
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
# c = max(contours, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(c)
# rect = cv2.rectangle(backup, (x,y), (x+w,y+h), (0,0,255),2)
# backup = orig_img[y:y+h,x:x+w]

# cv2.imwrite(storepath+'final.png',rect)

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
backup = orig_image[y:y+h,x:x+w]
backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LANCZOS4)
cv2.imwrite(storepath+'final_LANCZOS4.png',backup)
