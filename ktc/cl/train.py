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
from sklearn.metrics import auc, roc_curve
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
folds_string = 'allSubjectPaths{}.yaml'
def train(
    whichos,
    modalities,
    method,
    network,
    config,
    max_steps,
):
    '''
    Train a model with specified configs.
    Stores the cmdline args and other configs in one file for future ref, trains the model, stores results in csv and plots too.

    Args:
        whichos: operation system linux/windows/remote
        modalities (list[str]): the modalites being used
        max_steps (int): max training steps
        method: CT or MRI or both
        network: which network to use
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
    '''
    config = load.load_config(config)
    modalities = sorted(modalities, reverse=False)
    print("Modalities: %s"%modalities)
    print("Operating System: {}".format(whichos))
    print("Method: %s"%method)
    #network = str(network[0])

    metrics_file_name = config['data_options']['metrics_file_name']
    oldSavePath = os.path.join(config['data_options'][whichos]['save_path'],network)
    save_path = os.path.join(config['data_options'][whichos]['save_path'],metrics_file_name,'_'.join(modalities))
    oldDataPath = config['data_options'][whichos]['data_path']
    data_path = os.path.join(config['data_options'][whichos]['data_path'],'_'.join(modalities))
    split_CTMRI = config['data_options']['split_CTMRI']
    cv = int(config['data_options']['cv'])
    dump.dump_options(
        os.path.join(save_path, 'options_'+network+'_{}CV.yaml'.format(cv)),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )
    print("Data Path: {}".format(data_path))
    print("Save Path: {}".format(save_path))

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
    batch_size = config['data_options']['train']['batch_size']

    fold_acc = []
    here = cv
    #here = 1
    for i in range(here):
        send_path = os.path.join(data_path,'{}CV'.format(cv),folds_string.format(i))
        trainDS = dataset.train_ds(send_path, modalities, **config['data_options']['train'])
        testDS = dataset.predict_ds(send_path, modalities, **config['data_options']['test'])

        cvFold = str(i)
        save_models_here = os.path.join(save_path,'{}CV'.format(cv))
        weights_filename = os.path.join(save_models_here,'Fold'+cvFold,'weights/')
        preds_filename = os.path.join(save_models_here,'Fold'+cvFold,'predictions/')
        testdata_filename = os.path.join(save_models_here, 'Fold'+cvFold,'testdata/')
        os.makedirs(weights_filename, exist_ok=True)
        os.makedirs(preds_filename, exist_ok=True)
        os.makedirs(testdata_filename, exist_ok=True)
        tf.keras.backend.clear_session()

        num_neurons = 2
        dim = (224,224,3)
        n_trainsteps = folders.count_total(send_path,'train')//batch_size
        if network == 'cnn':
            model = vanillacnn.CNN(classifier_activation='softmax',num_classes=num_neurons)
        elif network == 'linearcombicnn':
            model = vanillacnn.linearCombiCNN(classifier_activation='softmax',num_classes=num_neurons)
        elif network == 'oldcnn':
            model = vanillacnn.oldCNN(classifier_activation='softmax',num_classes=num_neurons)
        elif network == 'vgg16':
            model = transfer_models.vgg16_net(classifier_activation='softmax')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=METRICS,
        )
        results = model.fit(
            trainDS,
            batch_size = batch_size,
            steps_per_epoch=n_trainsteps,
            epochs=max_steps,
            callbacks = [TqdmCallback(verbose=2)],
            verbose=0,
        )
        model.save_weights(weights_filename)
    
        x_test_numpy = []
        y_test_numpy = []
        for iter in testDS.as_numpy_iterator():
            x_test_numpy.append(iter[0])
            y_test_numpy.append(iter[1])

        x_test = np.array(x_test_numpy)
        y_test = np.array(y_test_numpy)
        np.save(testdata_filename+'X.npy',x_test)            
        np.save(testdata_filename+'y.npy',y_test)

        y_pred = np.array(model.predict(testDS))
        np.save(preds_filename+'yhat.npy',y_pred)
        print('Saved into: %s'%weights_filename)
        
        
        tp, fp, tn, fn = perf_measure(y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
        acc = (tp+tn) / (tp+fp+tn+fn)
        print("test acc: ",acc)           
        fold_acc.append(acc)
        nf_aml,nf_cc = folders.count_fromFiles(send_path, 'test')
        if os.path.isdir(save_path) and os.path.exists(save_path):
            sendpath = os.path.join(save_models_here,'Fold'+cvFold)
            os.makedirs(sendpath, exist_ok=True)
        colnames = ['Network','Modalities','Fold#','#AML(no)','#CCRCC(yes)','AUC','TP','FP','TN','FN','recall','specificity','f2','accuracy']
        # y_numpy = []
        # for iter in testDS.as_numpy_iterator():
        #     y_numpy.append(iter[1])
        # y_numpy = y_numpy[0]
        # y_pred = model.predict(testDS)
        # y_pred_classes = y_pred.argmax(axis=-1)
        # y_pred = np.squeeze(y_pred)
        
        eval_metrics = {k:0 for k in colnames}
        roundoff = 3
        eval_metrics['Network'] = network
        eval_metrics['Modalities'] = ' '.join(modalities)
        eval_metrics['Fold#'] = i
        eval_metrics['#AML(no)'] = nf_aml
        eval_metrics['#CCRCC(yes)'] = nf_cc
        eval_metrics['TP'] = tp
        eval_metrics['FP'] = fp
        eval_metrics['TN'] = tn
        eval_metrics['FN'] = fn
        eval_metrics['accuracy'] = np.round_(acc,roundoff)
        eval_metrics['AUC'] = plot_roc(y_test.argmax(axis=-1),y_pred.argmax(axis=-1), sendpath)
        plot_loss_acc(results, sendpath, network=network)
        f2 = plot_confmat(tp, fp, tn, fn, sendpath, roundoff)
        eval_metrics['f2'] = np.round_(f2,roundoff)
        eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
        eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)

        print(eval_metrics)

        metrics_path = os.path.join(save_path,'metrics_'+metrics_file_name+'.csv')
        if not os.path.exists(metrics_path):
            df = pd.DataFrame(columns=colnames)
            df = df.append(eval_metrics,ignore_index=True)
            df.to_csv(metrics_path)
        else:
            extra = pd.DataFrame([eval_metrics])
            extra.to_csv(metrics_path, mode='a', header=False)
        print("{} ***********************************RUN DONE FOLD***********************************".format(i))
        del model
        del results
    
    avg_acc = np.array(fold_acc).mean()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("AVG ACC: {}".format(avg_acc))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    
    return
    
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    #print("y_actual: ",y_actual, len(y_actual))
    #print("y_hat: ",y_hat, len(y_hat))
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

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
    return auc(fpr, tpr)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

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
    #plt.plot(req['val_'+metric])
    plt.title(network+' fine tuning model '+metric)
    plt.xlabel('epoch')
    #plt.legend(['train','val'], loc= 'upper left')
    plt.legend(['train'], loc= 'upper left')
    plt.savefig(os.path.join(path,metric+'.png'))
    plt.close(fig)

    #plot training and val accuracy
    metric = 'accuracy'
    fig2 = plt.figure()
    plt.plot(req[metric])
    #plt.plot(req['val_'+metric])
    plt.title(network+' fine tuning model '+metric)
    plt.xlabel('epoch')
    #plt.legend(['train','val'], loc= 'upper left')
    plt.legend(['train'], loc= 'upper left')
    plt.savefig(os.path.join(path,metric+'.png'))
    plt.close(fig2)

