import os
from numpy.lib.function_base import diff
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

backup = orig_image[y1:y2,x1:x2]
backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
cv2.imwrite(storepath+'_cropped_new_0.3.png',backup)


