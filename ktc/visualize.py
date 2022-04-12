import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.keras.layers.advanced_activations import Softmax

class my_model(Model):
    def __init__(self, dim):
        super(my_model, self).__init__()
        # self.Base  = tf.keras.applications.VGG16(input_shape=(dim), include_top = False, weights = 'imagenet')
        # self.GAP   = layers.GlobalAveragePooling2D()
        # self.BAT   = layers.BatchNormalization()
        # self.DROP  = layers.Dropout(rate=0.1)
        # self.DENS  = layers.Dense(256, activation='relu', name = 'dense_A')
        # self.OUT   = layers.Dense(1, activation='sigmoid')
        activation = 'relu'
        num_classes = 2
        classifier_activation = 'softmax'
        self.num_neurons = 224
        self.num_phases = 3
        self.linearphase1 = layers.Dense(224, activation=activation)
        self.linearphase2 = layers.Dense(224, activation=activation)
        self.linearphase3 = layers.Dense(224, activation=activation)

        self.compressNeuronPhase1 = layers.Dense(1, activation=activation)
        self.compressNeuronPhase2 = layers.Dense(1, activation=activation)
        self.compressNeuronPhase3 = layers.Dense(1, activation=activation)
        self.concat = layers.Concatenate()
        self.middleLayer = layers.Dense(3, activation=activation)
        self.linearchannel1 = layers.Dense(224*224, activation=activation)
        self.linearchannel2 = layers.Dense(224*224, activation=activation)
        self.linearchannel3 = layers.Dense(224*224, activation=activation)

        self.linearupsample = layers.Dense(self.num_neurons*self.num_neurons*3, activation=activation)
        self.reshape = layers.Reshape((self.num_neurons,self.num_neurons,3))

        self.conv1 = layers.Conv2D(64, 5, strides=2, activation=activation)

        self.conv2 = layers.Conv2D(128, 5, strides=2, activation=activation)
        self.drop  = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(256, 3, strides=2, activation=activation)
        self.drop  = layers.Dropout(0.5)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64)
        self.classify = layers.Dense(num_classes, activation=classifier_activation)

        self.lambdaa1 = layers.Lambda(lambda x: x[:,:,:,0], input_shape=(self.num_neurons, self.num_neurons, self.num_phases))
        self.lambdaa2 = layers.Lambda(lambda x: x[:,:,:,1:2][...,None], input_shape=(self.num_neurons, self.num_neurons, self.num_phases))
        self.lambdaa3 = layers.Lambda(lambda x: x[:,:,:,2][...,None], input_shape=(self.num_neurons, self.num_neurons, self.num_phases))
    
    def filter_layer(self, input, index):
        x = input[:,:,:,index]
        x = tf.expand_dims(x, axis=3)
        #print(x.shape)
        return x
    
    def call(self, input_tensor):
        # x  = self.Base(inputs)
        # g  = self.GAP(x)
        # b  = self.BAT(g)
        # d  = self.DROP(b)
        # d  = self.DENS(d)
        # return self.OUT(d)
        #middle = self.middleLayer(self.concat([neuron0, neuron1, neuron2]))
        
        input0 = self.filter_layer(input_tensor,0)
        neuron0 = self.compressNeuronPhase1(input0)
        #neuron0 = self.compressNeuronPhase2(neuron0)
        channel0 = self.flatten(neuron0)
        #print("input 0, neuron0, channel0: ", input0.shape, neuron0.shape, channel0.shape)
        
        input1 = self.filter_layer(input_tensor,1)
        neuron1 = self.compressNeuronPhase2(input1)
        channel1 = self.flatten(neuron1)
        
        input2 = self.filter_layer(input_tensor,2)
        neuron2 = self.compressNeuronPhase3(input2)
        channel2 = self.flatten(neuron2)

        x = self.concat([channel0, channel1, channel2])
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.drop(x)
        x = self.classify(x)
        return x
    
    # AFAIK: The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        x = layers.Input(shape=(dim))
        return Model(inputs=[x], outputs=self.call(x))

dim = (224,224,3)
model = my_model((dim))
model.build((None, *dim))
print(model.build_graph().summary())
# Just showing all possible argument for newcomer.  
tf.keras.utils.plot_model(
    model.build_graph(),                      # here is the trick (for now)
    to_file='model.png', dpi=96,              # saving  
    show_shapes=True, show_layer_names=True,  # show shapes and layer name
    expand_nested=False                       # will show nested block
)