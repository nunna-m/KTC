'''
Transfer Learnt Models defined here
'''

# built-in
import pdb
import numpy as np

# external
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras.backend import dtype
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
        self.rgb = 3
        self.z = np.zeros((1,224,224,1))
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False,weights='imagenet')
        self.base_model.trainable = False
        self.repeat = tf.keras.layers.Lambda(lambda x: self.repeat_const(x, self.z))
        self.concat = layers.Concatenate(axis=-1)
        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.prediction_layer = layers.Dense(1)
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
        
    def repeat_const(self, tensor, myconst):
        shapes = tf.shape(tensor)
        return tf.repeat(myconst, shapes[0], axis=0)
    
    @tf.function
    def call(self, input, training=False):
        xx = self.repeat(input)
        x = self.concat([input,xx])
        x = self.rescale(x)
        x = self.base_model(input_tensor=x)
        x = self.global_average_layer(x)
        x = self.dropout(x)
        x = self.prediction_layer(x)
        return x
