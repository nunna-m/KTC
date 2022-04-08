'''
CNN
'''

# built-in
import pdb

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras.layers.advanced_activations import Softmax
# customs
from . import components

class CNN(Model):
    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=None,
        dropout=None,
        trainable=True,
        activation='relu',
        classifier_activation='softmax',
        num_classes=2,
        **kargs,
    ):
        super().__init__(**kargs)
         # define all layers in init
        activation=layers.LeakyReLU()
        # Layer of Block 1
        self.conv1 = layers.Conv2D(64, 5, strides=2, activation=activation)
        self.max1  = layers.MaxPooling2D(3)
        self.bn1   = layers.BatchNormalization()

        # Layer of Block 2
        self.conv2 = layers.Conv2D(128, 5, strides=2, activation=activation)
        self.bn2   = layers.BatchNormalization()
        self.drop  = layers.Dropout(0.3)

        # Layer of Block 3
        self.conv3 = layers.Conv2D(256, 3, strides=2, activation=activation)
        self.bn2   = layers.BatchNormalization()
        self.drop  = layers.Dropout(0.5)

        self.flatten = layers.Flatten()
        # GAP, followed by Classifier
        self.gap   = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(64)
        self.classify = layers.Dense(num_classes, activation=classifier_activation)
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.drop(x)
        x = self.classify(x)
        return x
    

class oldCNN(Model):
    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=None,
        dropout=None,
        trainable=True,
        activation='relu',
        classifier_activation='softmax',
        num_classes=1,
        **kargs,
    ):
        super().__init__(**kargs)
         # define all layers in init
        activation=layers.LeakyReLU()
        # Layer of Block 1
        self.conv1 = layers.Conv2D(32, 3, strides=2, activation=activation)
        self.max1  = layers.MaxPooling2D(3)
        self.bn1   = layers.BatchNormalization()

        # Layer of Block 2
        self.conv2 = layers.Conv2D(64, 3, activation=activation)
        self.bn2   = layers.BatchNormalization()
        self.drop  = layers.Dropout(0.3)

        # GAP, followed by Classifier
        self.gap   = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes, activation=classifier_activation)
    
    @tf.function
    def call(self, input_tensor, training=False):
        # forward pass: block 1 
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.bn1(x)

        # forward pass: block 2 
        x = self.conv2(x)
        x = self.bn2(x)

        # droput followed by gap and classifier
        x = self.drop(x)
        x = self.gap(x)
        x = self.dense(x)
        return x