'''
abstraction for tf.data.Dataset API
'''
import os
import tensorflow as tf
import numpy as np
import glob

def train_dataset(
    traindir,
    batch_size,
    buffer_size,
    repeat = True,
    modalities=('am','tm','dc','ec','pc'),
    output_size=(224,224),
    aug_configs=None,
):
    
    dataset = load_raw(
        traindir,
        modalities=modalities,
        output_size=output_size,
    )

    dataset = custom_augmentation(
        dataset,
        aug_configs,
    )

    dataset = image_label(dataset, modalities=modalities)
    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat
    )

    return dataset

def load_raw(traindir, modalities=('am','tm','dc','ec','pc'), output_size=(224,224)):
    

    return dataset

def custom_augmentation(dataset, aug_configs):
    return dataset 

def image_label(dataset, modalities=('am','tm','dc','ec','pc')):
    return dataset

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    return dataset 

