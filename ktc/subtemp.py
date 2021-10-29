import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import glob
from functools import partial
import cv2
import sys
AUTOTUNE = tf.data.experimental.AUTOTUNE

imgpath = '/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/CCRCC/71335481/am/1.png'
labelpath = '/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/CCRCC/71335481/amL/1.png'
storepath = '/home/maanvi/Desktop/working/'
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
width_thresh = (orig_width//2)
height_thresh = (orig_height//2)

# if (x<width_thresh) and y<(height_thresh):
#     #tumor region in top left quadrant
#     backup = orig_image[y:y+224,x:x+224]
# elif (x>=width_thresh) and y<(height_thresh):
#     #tumor region in top right quadrant
#     diff = (width_thresh-x)//2
#     backup = orig_image[y:y+224,x-(224//2):x+()]

backup = orig_image[y:y+h,x:x+w]
backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
cv2.imwrite(storepath+'_cropped.png',backup)


