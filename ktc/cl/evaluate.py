'''
CLI for evaluate command, basically testing the model
'''

import os
import yaml

def evaluate(
    argsfile_json
):
    '''
    Evaluate on test data with supplied configurations.

    Args:
        argsfile_json: arguments specified inside this JSON file listed below
            configfiles (list[str]): config files absolute paths
            path_store_results: path to store weights/results
            path_data: path to dataset root
            max_epochs (int): max epochs
            early_stop: number of epochs to train without improvements in metrics.
                None(default): to disable the feature
            checkpoint_freq: interval of checkpoints
                default: 500 steps
            validate (bool): validate model on validation dataset
            path_val_data: path to validation data
    '''