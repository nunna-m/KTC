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
index = 0
img = cv2.imread(imgpath)[:,:,index]
label_img = cv2.imread(labelpath)[:,:,index]
cv2.imwrite(storepath+'image_{}.png'.format(index),img)
# cv2.imwrite(storepath+'label_image_{}.png'.format(index), label_img)
(orig_height, orig_width) = img.shape
image = cv2.imread(labelpath)[:,:,index]
image = cv2.resize(image, (orig_width, orig_height))
zero_mask = np.ones(image.shape[:2], dtype=image.dtype)
zero_mask  = cv2.bitwise_not(zero_mask)
result = cv2.bitwise_and(image, image, mask=zero_mask)
cv2.imwrite(storepath+'and.png',result)
backup = image.copy()
# lower_red = np.array([220])
# upper_red = np.array([255])
lower_red = np.array([0], np.uint8)
upper_red = np.array([0], np.uint8)
mask = cv2.inRange(image, lower_red, upper_red)
cv2.imwrite(storepath+'mask.png',mask)
#ret, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
img = img[y:y+h,x:x+w]
cv2.imwrite(storepath+'cropped.png',img)


