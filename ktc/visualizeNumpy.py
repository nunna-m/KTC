import matplotlib.pyplot
import os
import numpy as np
import yaml
import cv2

path = '/home/maanvi/LAB/Datasets/kt_combined/dc_ec_tm/numpyData/fullImage/train/5CV/0/15630104_AML_1.npz'

parts = path.rsplit(os.path.sep,7)
modalities = parts[1].split('_')
subjectID,clas,_ = parts[7].split('_')

data = np.load(path)
print(list(data.keys()))
#print(f"image: {data['image']} and label:{data['label']}")
image = data['image']
label = data['label']

tempPath = '/home/maanvi/Desktop/vis'
os.makedirs(tempPath,exist_ok=True)

for i,mod in enumerate(modalities):
    cv2.imwrite(os.path.join(tempPath,f'{mod}.png'),image[:,:,i])