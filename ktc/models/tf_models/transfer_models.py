'''
Transfer Learnt Models defined here
'''

# built-in
import pdb
import numpy as np
from numpy.lib.financial import rate

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras.backend import dropout, dtype
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.keras import backend as K
# customs
from . import components

class mobile_net(Model):
    def __init__(
        self,
        size=224,
        **kargs,
    ):
        super().__init__(**kargs)

        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        #self.base_model.trainable = False
        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.prediction_layer = layers.Dense(1)
        
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.rescale(x)
        x = self.base_model(x)
        x = self.global_average_layer(x)
        x = self.dropout(x)
        x = self.prediction_layer(x)
        return x


class alex_net(Model):
    def __init__(
        self,
        padding='same',
        activation='relu',
        dropout=0.5,
        **kargs,
    ):
        super().__init__(**kargs)
        self.conv1 = layers.Conv2D(96, 11, strides=4, activation=activation)
        self.bn1   = layers.BatchNormalization()
        self.max1  = layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv2 = layers.Conv2D(256, 5, strides=1, activation=activation, padding=padding)
        self.bn2   = layers.BatchNormalization()
        self.max2  = layers.MaxPooling2D(pool_size=3, strides=2)

        self.conv3 = layers.Conv2D(384, 3, strides=1,activation=activation, padding=padding)
        self.bn3   = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(384, 3, strides=1,activation=activation, padding=padding)
        self.bn4   = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(256, 3, strides=1,activation=activation, padding=padding)
        self.bn5   = layers.BatchNormalization()

        self.max3  = layers.MaxPooling2D(pool_size=3, strides=2)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation=activation)
        self.dropout1 = layers.Dropout(rate=dropout)
        
        self.dense2 = layers.Dense(4096, activation=activation)
        self.dropout2 = layers.Dropout(rate=dropout)
        
        self.dense3 = layers.Dense(1, activation='sigmoid')
    
    @tf.function
    def call(self, input_tensor, training=False):
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.max3(x)

        x = self.flatten(x)
        x= self.dense1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.dropout2(x)

        x = self.dense3(x)

        return x

class vgg16_net(Model):
    def __init__(
        self,
        activation='relu',
        classifier_neurons=1,
        **kargs,
    ):
        super().__init__(**kargs)
        self.base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        for self.layer in self.base_model.layers[:15]:
            self.layer.trainable = False
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation=activation)
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation=activation)
        self.dense3 = layers.Dense(classifier_neurons, activation='sigmoid')
        
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = self.base_model(input_tensor)
        x = self.flatten(x)
        x = self.dense1(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x