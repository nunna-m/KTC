'''
CLI for train command
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ktc.utils import get, store, load, dump
from ktc import dataset

'''
interface for training models
'''

# built-in
from datetime import datetime

# external
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from tqdm.keras import TqdmCallback
import pandas as pd
import csv
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
    fold,
    network,
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
        fold (int): the fold representing random creation of train/val/test
        network (list[str]): which neural network to use
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
    split_CTMRI = config['data_options']['split_CTMRI']
   # cv = config['data_options']['cv']
    fold = str(fold)
    network = str(network[0])
    
    data_path = os.path.join(data_path, 'fold'+fold)
    save_path = os.path.join(save_path, 'fold'+fold, network)
    

    dump.dump_options(
        os.path.join(save_path, 'options_fold'+fold+'_'+network+'.yaml'),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )
    print(save_path,data_path)
    
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
    tf.keras.backend.clear_session()
    num_neurons = 1
    #network = 'cnn'
    if network == 'alexnet':
        model = transfer_models.alex_net(classifier_neurons=num_neurons)
    if network == 'vgg16':
        model = transfer_models.vgg16_net(classifier_neurons=num_neurons)
    if network == 'resnet':
        model = transfer_models.res_net50(classifier_neurons=num_neurons)
    if network == 'cnn':
        model = vanillacnn.CNN(activation='relu',num_classes=1)
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
      keras.metrics.AUC(name='prc', curve='PR'), 
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    batch_size = config['data_options']['train']['batch_size']
    n_trainsteps = folders.count_samples(modalities,data_path,'train')['total']//batch_size
    if validate:
        n_valsteps = folders.count_samples(modalities,data_path,'val')['total']//batch_size
        #n_trainsteps = 1
        print("batchsize, trainsteps, valsteps")
        print(batch_size, n_trainsteps, n_valsteps)
        results = model.fit(
        ds,
        batch_size = batch_size,
        validation_data=val_ds,
        steps_per_epoch=n_trainsteps,
        epochs=max_steps,
        callbacks = [reduce_lr, TqdmCallback(verbose=2)],
        verbose=0,
    )
    else:
        results = model.fit(
        ds,
        batch_size = batch_size,
        steps_per_epoch=n_trainsteps,
        epochs=max_steps,
        callbacks = [reduce_lr, TqdmCallback(verbose=2)],
        verbose=0,
    )

    #predict
    test_ds = dataset.predict_ds(data_path, modalities, **config['data_options']['test'])

    #print(list(test_ds.as_numpy_iterator()))

    loss, tp, fp, tn, fn, acc, precision, recall, AUC, prc = model.evaluate(test_ds)
    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)
    print("test loss, test acc: ",model.evaluate(test_ds))
    print("{} ***********************************RUN DONE ***********************************".format(modalities))

    nf_aml = folders.count_samples(modalities,data_path,'test')['AML']
    nf_cc = folders.count_samples(modalities,data_path,'test')['CCRCC']
    if os.path.isdir(save_path) and os.path.exists(save_path):
        sendpath = os.path.join(save_path, '_'.join(modalities))
        os.makedirs(sendpath, exist_ok=True)
    colnames = ['Modalities','#AML(no)','#CCRCC(yes)','AUC','TP','FP','TN','FN','recall','specificity','f2','accuracy']
    y_numpy = []
    for iter in test_ds.as_numpy_iterator():
        y_numpy.append(iter[1])
    y_numpy = y_numpy[0]
    y_pred = model.predict(test_ds)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_pred = np.squeeze(y_pred)
    print(y_pred_classes)
    
    eval_metrics = {k:0 for k in colnames}
    roundoff = 3
    eval_metrics['Modalities'] = ' '.join(modalities)
    eval_metrics['#AML(no)'] = nf_aml
    eval_metrics['#CCRCC(yes)'] = nf_cc
    eval_metrics['TP'] = tp
    eval_metrics['FP'] = fp
    eval_metrics['TN'] = tn
    eval_metrics['FN'] = fn
    eval_metrics['accuracy'] = np.round_(acc,roundoff)
    eval_metrics['AUC'] = AUC
    plot_loss_acc(results, sendpath, network=network)
    plot_roc(y_numpy, y_pred, sendpath)
    f2 = plot_confmat(tp, fp, tn, fn, sendpath, roundoff)
    eval_metrics['f2'] = np.round_(f2,roundoff)
    eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
    eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)

    print(eval_metrics)

    metrics_path = os.path.join(save_path,'metrics.csv')
    if not os.path.exists(metrics_path):
        df = pd.DataFrame(columns=colnames)
        df = df.append(eval_metrics,ignore_index=True)
        df.to_csv(metrics_path)
    else:
        extra = pd.DataFrame([eval_metrics])
        extra.to_csv(metrics_path, mode='a', header=False)
    
    return results
    

def plot_roc(y_true, y_pred, path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig(os.path.join(path,'roc.png'))
    plt.close(fig)
    
def fbeta(tp, fp, tn, fn, beta=2.0):
    squared = pow(beta, 2)
    numerator = (1 + squared)*tp
    denominator = ((1 + squared)*tp) + squared*fn + fp
    return numerator/denominator

def plot_confmat(tp, fp, tn, fn, path, roundoff, beta=2.0):
    f2 = fbeta(tp, fp, tn, fn, beta)
    print('f2-score: ',f2)
    cf = np.array([[tp,fn],
            [fp, tn]])
    print(cf)

    classes = ['CCRCC','AML']
    title = "Confusion Matrix"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cf)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks([0,0,1])
    ax.set_yticks([0,0,1])
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(os.path.join(path,'cf.png'))
    plt.close(fig)

    return f2

def plot_loss_acc(history, path, network):
    req = history.history
    
    #plot training and val loss
    metric = 'loss'
    fig = plt.figure()
    plt.plot(req[metric])
    plt.plot(req['val_'+metric])
    plt.title(network+' fine tuning model '+metric)
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc= 'upper left')
    plt.savefig(os.path.join(path,metric+'.png'))
    plt.close(fig)

    #plot training and val accuracy
    metric = 'accuracy'
    fig2 = plt.figure()
    plt.plot(req[metric])
    plt.plot(req['val_'+metric])
    plt.title(network+' fine tuning model '+metric)
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc= 'upper left')
    plt.savefig(os.path.join(path,metric+'.png'))
    plt.close(fig2)

