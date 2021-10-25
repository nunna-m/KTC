import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.keras.applications import vgg16, ResNet50


vgg = vgg16.VGG16(weights='imagenet',include_top=False)
print(vgg.summary())