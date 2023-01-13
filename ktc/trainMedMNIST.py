#sample command: python trainMedMNIST.py /home/maanvi/LAB/code/organamnist.npz /home/maanvi/LAB/pre_trained_models/vgg16_plain/cp.ckpt
import cv2
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import MaxPool2D

def VGG16(num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu",name="block1_conv1"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu",name="block1_conv2"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="block1_pool"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name="block2_conv1"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name="block2_conv2"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="block2_pool"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name="block3_conv1"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name="block3_conv2"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name="block3_conv3"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="block3_pool"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block4_conv1"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block4_conv2"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block4_conv3"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="block4_pool"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block5_conv1"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block5_conv2"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name="block5_conv3"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="block5_pool"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    return model

checkpoint_path = sys.argv[2]
dataset_path = sys.argv[1]
data = np.load(dataset_path, allow_pickle=True)
#print(list(data.keys()))
train_data = data['train_images']
train_labels = data['train_labels']
class_labels = list(np.unique(data['train_labels']))
train_labels = to_categorical(train_labels, len(class_labels))
#print(f'Train data shape: {train_data.shape}')
val_data = data['val_images']
val_labels = data['val_labels']
val_labels = to_categorical(val_labels,len(class_labels))
#print(f'Val data shape: {val_data.shape}')
test_data = data['test_images']
test_labels = data['test_labels']
test_labels = to_categorical(test_labels,len(class_labels))
#print(f'Test data shape: {test_data.shape}')

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (224,224))
    return image, label

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

print('Using GPU---------')
print(tf.config.list_physical_devices('GPU'))
print('*********************')
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.map(format_example)
train_dataset = train_dataset.map(lambda x,y:(normalization_layer(x),y))

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.map(format_example)
val_dataset = val_dataset.map(lambda x,y:(normalization_layer(x),y))

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.map(format_example)
test_dataset = test_dataset.map(lambda x,y:(normalization_layer(x),y))

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

for image_batch, label_batch in train_dataset.take(2):
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = VGG16(num_classes=len(class_labels))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])



checkpoint_dir = os.path.dirname(checkpoint_path)

#Create a callback that saves the model weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_dataset, 
          validation_data=val_dataset,
          epochs=100,
          callbacks=[cp_callback])

# vgg_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3))
# for layer in model.layers:
#     print(layer.name)