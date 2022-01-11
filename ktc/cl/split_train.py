'''
CLI for train command
'''

import os

from numpy.lib.npyio import save
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

def split_train(
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
    if split_CTMRI:
        ct_modalities = [modal for modal in modalities if 'c' in modal]
        ct_trainds = dataset.train_ds(data_path, ct_modalities, **config['data_options']['train'])
        ct_valds = dataset.eval_ds(data_path, ct_modalities, **config['data_options']['eval'])
        ct_testds = dataset.predict_ds(data_path, ct_modalities, **config['data_options']['test'])

        tf.keras.backend.clear_session()
        num_neurons = 1
        base_learning_rate = 0.00001
        decay = base_learning_rate / max_steps
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=base_learning_rate)
        batch_size = config['data_options']['train']['batch_size']
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
        

        # ct_model = transfer_models.vgg16_net(classifier_neurons=num_neurons)
        # ct_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        #         loss=tf.keras.losses.BinaryCrossentropy(),
        #         metrics=METRICS)
        
        # ct_n_trainsteps = folders.count_samples(ct_modalities,data_path,'train')['total']//batch_size
        # ct_n_valsteps = folders.count_samples(ct_modalities,data_path,'val')['total']//batch_size
        # print("batchsize, trainsteps, valsteps")
        # print(batch_size, ct_n_trainsteps, ct_n_valsteps)

        # ct_results = ct_model.fit(ct_trainds,
        # batch_size = batch_size,
        # validation_data=ct_valds,
        # steps_per_epoch=ct_n_trainsteps,
        # epochs=max_steps,
        # callbacks = [reduce_lr, TqdmCallback(verbose=2)],
        # verbose=0,
        # )
        
        # save_results_and_plot(model=ct_model, testData=ct_testds,modalities=ct_modalities, dataPath=data_path, savePath=save_path, fit_results=ct_results, network=network)



        mri_modalities = [modal for modal in modalities if 'm' in modal]
        mri_trainds = dataset.train_ds(data_path, mri_modalities, **config['data_options']['train'])
        mri_valds = dataset.eval_ds(data_path, mri_modalities, **config['data_options']['eval'])
        mri_testds = dataset.predict_ds(data_path, mri_modalities, **config['data_options']['test'])
        mri_model = transfer_models.vgg16_net(classifier_neurons=num_neurons)
        mri_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)

        mri_n_trainsteps = folders.count_samples(mri_modalities,data_path,'train')['total']//batch_size
        mri_n_valsteps = folders.count_samples(mri_modalities,data_path,'val')['total']//batch_size
        print("batchsize, trainsteps, valsteps")
        print(batch_size, mri_n_trainsteps, mri_n_valsteps)

        mri_results = mri_model.fit(mri_trainds,
        batch_size = batch_size,
        validation_data=mri_valds,
        steps_per_epoch=mri_n_trainsteps,
        epochs=max_steps,
        callbacks = [reduce_lr, TqdmCallback(verbose=2)],
        verbose=0,
        )
        
        save_results_and_plot(model=mri_model, testData=mri_testds,modalities=mri_modalities, dataPath=data_path, savePath=save_path, fit_results=mri_results, network=network)
        


    return mri_results

def save_results_and_plot(model, testData, modalities, dataPath, savePath, fit_results, network):
    loss, tp, fp, tn, fn, acc, precision, recall, AUC, prc = model.evaluate(testData)
    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)
    print("test loss, test acc: ",model.evaluate(testData))
    print("{} ***********************************RUN DONE ***********************************".format(modalities))

    nf_aml = folders.count_samples(modalities,dataPath,'test')['AML']
    nf_cc = folders.count_samples(modalities,dataPath,'test')['CCRCC']
    if os.path.isdir(savePath) and os.path.exists(savePath):
        sendpath = os.path.join(savePath,'stacked_level0', '_'.join(modalities))
        os.makedirs(sendpath, exist_ok=True)
    colnames = ['Modalities','#AML(no)','#CCRCC(yes)','AUC','TP','FP','TN','FN','recall','specificity','f2','accuracy']
    y_numpy = []
    for iter in testData.as_numpy_iterator():
        y_numpy.append(iter[1])
    y_numpy = y_numpy[0]
    y_pred = model.predict(testData)
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
    plot_loss_acc(fit_results, sendpath, network=network)
    plot_roc(y_numpy, y_pred, sendpath)
    f2 = plot_confmat(tp, fp, tn, fn, sendpath, roundoff)
    eval_metrics['f2'] = np.round_(f2,roundoff)
    eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
    eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)

    print(eval_metrics)

    metrics_path = os.path.join(savePath,'stacked_level0','metrics.csv')
    if not os.path.exists(metrics_path):
        df = pd.DataFrame(columns=colnames)
        df = df.append(eval_metrics,ignore_index=True)
        df.to_csv(metrics_path)
    else:
        extra = pd.DataFrame([eval_metrics])
        extra.to_csv(metrics_path, mode='a', header=False)

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

