'''
CLI for train command
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from ktc.utils import get, store, load, dump
from ktc import dataset
from ktc.cl import train as trainfile

'''
interface for training models
'''

# built-in
from datetime import datetime

# external
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, fbeta_score, precision_score, recall_score, accuracy_score
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import numpy as np
# customs
from ktc.utils import get, store, load, dump
from ktc import dataset, folders
from ktc.models.tf_models import transfer_models, vanillacnn

from tensorflow.python.keras.applications.vgg16 import VGG16

logsdir = "logs/fit/transfer_learning/" + datetime.now().strftime("%m%d-%H%M")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logsdir)

def train_stacked(
    whichos,
    config,
    modalities,
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
        whichos: operating system linux/windows/remote(docker)
        config (list[str]): configuration file path
            This option accepts arbitrary number of configs.
            If a list is specified, the first one is considered
            as a "main" config, and the other ones will overwrite the content
        modalities (list[str]): the modalites being used
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
    print("modalities and os: ",modalities, whichos)
    modalities = sorted(modalities, reverse=False)
    save_path = config['data_options'][whichos]['save_path']
    data_path = config['data_options'][whichos]['data_path']
    dump.dump_options(
        os.path.join(save_path, 'options.yaml'),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )
    print(save_path,data_path)
    # ct_modalities = [i for i in modalities if "c" in i]
    # mri_modalities = [i for i in modalities if "m" in i]
    # assert ct_modalities, mri_modalities
    ds = dataset.train_ds(data_path, modalities, **config['data_options']['train'])
    # ds_ct = dataset.train_ds(data_path, modalities=ct_modalities, **config['data_options']['train'])
    # ds_mri = dataset.train_ds(data_path, modalities=mri_modalities, **config['data_options']['train'])
    if validate:
        #assert val_data_path is not None
        val_ds = dataset.eval_ds(data_path, modalities, **config['data_options']['eval'])
        # val_ds_ct = dataset.eval_ds(data_path, modalities=ct_modalities, **config['data_options']['eval'])
        # val_ds_mri = dataset.eval_ds(data_path, modalities=mri_modalities, **config['data_options']['eval'])
    else:
        val_ds = None
        val_ds_ct = None
        val_ds_mri = None

    #loading testing dataset
    test_ds = dataset.predict_ds(data_path, modalities, **config['data_options']['test'])
    # test_ds_ct = dataset.predict_ds(data_path, modalities=ct_modalities, **config['data_options']['test'])
    # test_ds_mri = dataset.predict_ds(data_path, modalities=mri_modalities, **config['data_options']['test'])


    tf.keras.backend.clear_session()
    
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    tf.keras.backend.clear_session()
    num_neurons=1
    base_learning_rate = 0.00001
    batch_size = config['data_options']['train']['batch_size']
    n_trainsteps = folders.count_samples(modalities,data_path,'train')['total']//batch_size
    n_valsteps = folders.count_samples(modalities,data_path,'val')['total']//batch_size
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=base_learning_rate)
    
    
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_model.layers:
	    layer.trainable = False
    vgg_model.summary()  #Trainable parameters will be 0

    # vgg_model.fit(
    #     ds,
    #     epochs=1,
    # )
    feature_extractor = vgg_model.predict(ds)
    print(feature_extractor)
    # model = tfdf.keras.RandomForestModel()
    # model.compile(metrics=['accuracy'])
    # with sys.pipes():
    #     model.fit(x=ds)

    #Now, let us use features from convolutional network for RF
    # X_train, y_train = list(ds.as_numpy_iterator())
    # X_test, y_test = list(test_ds.as_numpy_iterator())
    # feature_extractor=vgg_model.predict(X_train)
    # print(feature_X_train.shape, y_train.shape)
    # print(X_test.shape, y_train.shape)
    # print("done extracting features from vgg model")
    
    # features = feature_X_train.reshape(feature_extractor.shape[0], -1)

    # X_for_training = features #This is our X input to RF
    
    # model = xgb.XGBClassifier()
    # model.fit(X_for_training, y_train, verbose=1)
    # print("done fitting on xgboost")

    # X_test_feature = vgg_model.predict(X_test)
    # X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

    # prediction = model.predict(X_test_features)
    # #Print overall accuracy
    # from sklearn import metrics
    # print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))

    # #Confusion Matrix - verify accuracy of each class
    # from sklearn.metrics import confusion_matrix

    # cm = confusion_matrix(y_test, prediction)
    # print(cm)
    
    return 0


#ct network
    ct_model = transfer_models.stackedGB_net(classifier_neurons=num_neurons)
    ct_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    results = ct_model.fit(
        ds_ct,
        batch_size = batch_size,
        validation_data=val_ds_ct,
        steps_per_epoch=n_trainsteps,
        epochs=max_steps,
        callbacks = [reduce_lr],
        verbose=1
    )
    ct_features = ct_model.predict(test_ds_ct)


    #mri network
    mri_model = transfer_models.stackedGB_net(classifier_neurons=num_neurons)
    mri_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    results = mri_model.fit(
        ds_mri,
        batch_size = batch_size,
        validation_data=val_ds_mri,
        steps_per_epoch=n_trainsteps,
        epochs=max_steps,
        callbacks = [reduce_lr],
        verbose=1
    )