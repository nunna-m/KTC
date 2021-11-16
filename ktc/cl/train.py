'''
CLI for train command
'''

import os
from tensorflow.keras import metrics
from tensorflow.python.keras.callbacks import LearningRateScheduler
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
from datetime import datetime

# external
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import dsargparse
import yaml
import tensorboard as tb
import numpy as np
# customs
from ktc.utils import get, store, load, dump
from ktc import dataset, folders
from ktc.models.tf_models import transfer_models, vanillacnn

logsdir = "logs/fit/transfer_learning/" + datetime.now().strftime("%m%d-%H%M")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logsdir)

def train(
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
    save_path = config['data_options'][whichos]['save_path']
    data_path = config['data_options'][whichos]['data_path']
    dump.dump_options(
        os.path.join(save_path, 'options.yaml'),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )
    ds = dataset.train_ds(data_path, modalities, **config['data_options']['train'])
    if validate:
        #assert val_data_path is not None
        val_ds = dataset.eval_ds(data_path, modalities, **config['data_options']['eval'])
    else: val_ds = None

    if visualize:
        visualization = {
            'train': dataset.eval_ds(data_path, modalities, **config['data_options']['eval'], include_meta=True),
            'validation': dataset.eval_ds(val_data_path, modalities,**config['data_options']['eval'], include_meta=True),
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
    num_neurons = 1
    model = transfer_models.vgg16_net(classifier_neurons=num_neurons)
    base_learning_rate = 0.00001
    decay = base_learning_rate / max_steps
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=base_learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    mode='min',
                    restore_best_weights=True)
    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1+decay*epoch)
    
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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    batch_size = config['data_options']['train']['batch_size']
    n_trainsteps = folders.count_samples(modalities,data_path,'train')['total']//batch_size
    n_valsteps = folders.count_samples(modalities,data_path,'val')['total']//batch_size
    results = model.fit(
        ds,
        batch_size = batch_size,
        validation_data=val_ds,
        steps_per_epoch=n_trainsteps,
        validation_steps=n_valsteps,
        epochs=max_steps,
        callbacks = [reduce_lr],
        verbose=1
        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],

    )
    
    # dump.dump_train_results(
    #     os.path.join(save_path, 'results.pkl'),
    #     results,
    #     format_='pickle',
    # )

    #predict
    test_ds = dataset.predict_ds(data_path, modalities, **config['data_options']['test'])

    print("test loss, test acc: ",model.evaluate(test_ds))
    print("{} ***********************************RUN DONE ***********************************".format(modalities))

    y_numpy = []
    for iter in test_ds.as_numpy_iterator():
        y_numpy.append(iter[1])
    y_numpy = np.array(y_numpy)
    y_pred = model.predict(test_ds)
    plot_loss_acc(results, save_path, modalities, metrics=METRICS)

    plot_roc(y_numpy, y_pred, modalities, save_path)
    
    return results

def plot_roc(y_true, y_pred, modals, path):
    if os.path.isdir(path) and os.path.exists(path):
        savepath = os.path.join(path, 'graphs','_'.join(modals))
        os.makedirs(savepath, exist_ok=True)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig = plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig(os.path.join(savepath,'roc.png'))
    plt.close(fig)

def plot_loss_acc(history, path, modals, metrics):
    req = history.history
    if os.path.isdir(path) and os.path.exists(path):
        savepath = os.path.join(path, 'graphs','_'.join(modals))
        os.makedirs(savepath, exist_ok=True)
    
    #plot training and val loss
    metric = 'loss'
    fig = plt.figure()
    plt.plot(req[metric])
    plt.plot(req['val_'+metric])
    plt.title('vgg16 fine tuning model '+metric)
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc= 'upper left')
    plt.savefig(os.path.join(savepath,metric+'.png'))
    plt.close(fig)

    #plot training and val accuracy
    metric = 'accuracy'
    fig2 = plt.figure()
    plt.plot(req[metric])
    plt.plot(req['val_'+metric])
    plt.title('vgg16 fine tuning model '+metric)
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc= 'upper left')
    plt.savefig(os.path.join(savepath,metric+'.png'))
    plt.close(fig2)

