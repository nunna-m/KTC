#incomplete
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the metrics curves and store them
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
    
    def on_epoch_end(self, epoch, logs={}):
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metric[metric] = [logs.get(metric)]
        
        metrics = [x for x in logs]