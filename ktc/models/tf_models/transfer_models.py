'''
Transfer Learnt Models defined here
'''

# built-in
import os
import tempfile
import pdb
import numpy as np

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

def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


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
        classifier_neurons=1,
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
        
        self.dense3 = layers.Dense(classifier_neurons, activation='sigmoid')
    
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
        #self.base_model = add_regularization(self.base_model)
        for self.layer in self.base_model.layers[:15]:
            self.layer.trainable = False
        for self.layer in self.base_model.layers[15:]:
            self.layer.trainable = True
        
        self.last_layer = self.base_model.get_layer('block5_pool')
        self.top_model = self.last_layer.output
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation=activation)
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(256, activation=activation)
        self.dense3 = layers.Dense(classifier_neurons, activation='sigmoid')
        
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = self.base_model(input_tensor)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x

class res_net50(Model):
    def __init__(
        self,
        activation='relu',
        classifier_neurons=1,
        **kargs,
    ):
        super().__init__(**kargs)
        self.base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        #self.base_model = add_regularization(self.base_model)
        for self.layer in self.base_model.layers[:26]:
            self.layer.trainable = False
        for self.layer in self.base_model.layers[26:]:
            self.layer.trainable = True
        
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation=activation)
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(256, activation=activation)
        self.dense3 = layers.Dense(classifier_neurons, activation='sigmoid')
        
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = self.base_model(input_tensor)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x


class stackedNet(Model):
    def __init__(
        self,
        activation='relu',
        classifier_neurons=1,
        **kargs,
    ):
        super().__init__(**kargs)
        self.base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        #self.base_model = add_regularization(self.base_model)
        for self.layer in self.base_model.layers:
            self.layer.trainable = False
        
        self.last_layer = self.base_model.get_layer('block5_pool')
        self.top_model = self.last_layer.output
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation=activation)
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(256, activation=activation, name='penultimate')
        self.dense3 = layers.Dense(classifier_neurons, activation='sigmoid')
        
    
    @tf.function
    def call(self, input_tensor, training=False):
        x = self.base_model(input_tensor)
        # x = self.gap(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = self.dropout(x)
        # x = self.dense3(x)
        return x