import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import MaxPool2D

data = np.load('/home/maanvi/LAB/code/organamnist.npz', allow_pickle=True)
print(list(data.keys()))
train_data = data['train_images']
print(f'Train data shape: {train_data.shape}')
val_data = data['val_images']
print(f'Val data shape: {val_data.shape}')
test_data = data['test_images']
print(f'Test data shape: {test_data.shape}')

class_labels = list(np.unique(data['train_labels']))
train_datagen = ImageDataGenerator(
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe