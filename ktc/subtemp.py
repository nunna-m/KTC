
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np


X = np.random.randint(0,10, (100,4,4,2))
Y = np.random.uniform(0,1, (100,20,5))

myarr = np.ones((1, 4,4,2)).astype('float32')
myconst = tf.convert_to_tensor(myarr)

def repeat_const(tensor, myconst):
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)

inputs = tf.keras.layers.Input((4,4,2))
#x = tf.keras.layers.Embedding(10,4,4,2)(inputs)
x=inputs
xx = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(x)
x = tf.keras.layers.Concatenate(axis=-1)([x, xx])
model = tf.keras.models.Model(inputs=inputs, outputs=x)
model.compile('adam', 'mse')
print(model.summary())
#model.fit(X, Y, epochs=3)