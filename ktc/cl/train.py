'''
CLI for train command
'''

import os
import yaml

from ktc.utils import get, store
from ktc import dataset

def train(
    path_data,
    path_save,
    filename_configs_save,
    configfilepaths,
    epochs,
    early_stop_count,
    checkpoint_freq,
    validate=False,
    path_val_data=None,
):
    '''
    Train on data with supplied configurations.

    Args:
        path_data: path to dataset root
        path_store: path to store weights/results
        configfilepaths (list[str]): config files absolute paths
        epochs (int): max epochs
        early_stop_count: number of epochs to train without improvements in metrics.
            None(default): to disable the feature
        checkpoint_freq: interval of checkpoints
            default: 500 steps
        validate (bool): validate model on validation dataset
        path_val_data: path to validation data
    '''

    #extract configs from the json file
    #get train_data using the tf.data.Dataset API interface functions using the path_data and other configs from configfiles
    #check bool args like validate
        #make sure path_val_data exists
        #get val_data again using the tf.data.Dataset API interface function using path_val_data and configs from the configfiles
    #define a model (use it from another file where model has been defined) and pass the extracted and required configs to the function
    #results = model.train() with extracted train_ds and other args like max_epochs, early_stop etc
    #store the train results in path_store_results (maybe pickle?)
    #return results?

    config = get.get_configs(configfilepaths)
    store.store_configs(
        os.path.join(path_save, filename_configs_save),
        config = config,
        path_save=path_save,
        path_data=path_data,
    )
    ds = dataset.train_dataset(path_data,config)
    

    
