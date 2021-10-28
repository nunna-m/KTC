'''
CLI for train command
'''

import os
import yaml

from ktc.utils import get, store, load, dump
from ktc import dataset

'''
interface for training models
'''

# built-in
import pdb
import os
import argparse

# external
import tensorflow as tf
from tensorflow import keras
import dsargparse
import yaml

# customs
from ktc.utils import get, store, load, dump
from ktc import dataset, engine
from ktc.models.tf_models import transfer_models, vanillacnn

def train(
    config,
    save_path,
    data_path,
    max_steps,
    early_stop_steps=None,
    save_freq=500,
    validate=False,
    val_data_path=None,
    visualize=False,
    profile=False,
):
    '''
    Train a model with specified configs.
    This function will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        config (list[str]): configuration file path
            This option accepts arbitrary number of configs.
            If a list is specified, the first one is considered
            as a "main" config, and the other ones will overwrite the content
        save_path: where to save weights/configs/results
        data_path (list[str]): path to the data root dir
        max_steps (int): max training steps
        early_stop_steps: steps to train without improvements
            None(default) disables this feature
        save_freq: interval of checkpoints
            default: 500 steps
        validate: also validate the model on the validation dataset
        val_data_path (list[str]): path to the validation dataset
        visualize (bool): should visualize results
        profile (bool): enable profilling
    '''
    config = load.load_config(config)
    dump.dump_options(
        os.path.join(save_path, 'options.yaml'),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )
    ds = dataset.train_ds(data_path, **config['data_options']['train'])
    if validate:
        #assert val_data_path is not None
        val_ds = dataset.eval_ds(data_path, **config['data_options']['eval'])
    else: val_ds = None

    if visualize:
        visualization = {
            'train': dataset.eval_ds(data_path, **config['data_options']['eval'], include_meta=True),
            'validation': dataset.eval_ds(val_data_path, **config['data_options']['eval'], include_meta=True),
        }
    else: visualization = {}

    #model = engine.TFKerasModel(config)
    # model = vanillacnn.CNN(activation='relu',num_classes=2)
    # model.compile(
    #     loss=tf.keras.losses.CategoricalCrossentropy(),
    #     metrics=tf.keras.metrics.CategoricalAccuracy(),
    #     optimizer=tf.keras.optimizers.Adam(),
    # )
    # results = model.fit(
    #     ds,
    #     validation_data=val_ds,
    #     steps_per_epoch=1,
    #     epochs=100,

    # )
    # print(results)
    model = transfer_models.mobile_net()
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=tf.keras.metrics.BinaryAccuracy())
    #input_shape = (None, 224, 224, 3)
    #model.build(input_shape)      
    #print(model.summary())
    results = model.fit(
        ds,
        validation_data=val_ds,
        steps_per_epoch=1,
        epochs=200,

    )
    print(results)

    return results

