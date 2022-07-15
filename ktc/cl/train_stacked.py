'''
interface to train stacked model
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
import pandas as pd
import numpy as np
from ktc.utils import load, dump, metrics
from ktc import dataset, folders
from ktc.models.tf_models import transfer_models, vanillacnn

folds_string = 'allSubjectPaths{}.yaml'

def train_stacked(
    whichos,
    method,
    network,
    config,
    max_steps,
    filename=None,
):
    '''
    Train a stacked model with the networks specified as level0 and level1 in the config file. Stores the cmdline args and other configs in one file for future ref, trains the model, stores results in csv and plots too.

    Args:
        whichos: operation system linux/windows/remote
        method: either CT or MRI
        network: which network to use (for level0 networks)
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
        filename: desired metrics filename, default is level0_level1_numberofepochs
        '''
    config = load.load_config(config)
    modalities = ['am', 'dc', 'ec', 'pc', 'tm']
    
    print("Modalities: %s"%modalities)
    print("Operating System: {}".format(whichos))
    print("Method: %s"%method)
    cv = int(config['data_options']['cv'])
    batch_size = config['data_options']['train']['batch_size']
    if not filename:
        metrics_file_name = '{}_{}eps'.format(network,max_steps)
    else:
        metrics_file_name = filename
    save_path = os.path.join(config['data_options'][whichos]['save_path'],metrics_file_name)
    data_path = os.path.join(config['data_options'][whichos]['data_path'],'_'.join(modalities))
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
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), 
    ]
    batch_size = config['data_options']['train']['batch_size']
    
    if method=='CT':
        letter='c'
    elif method=='MRI':
        letter='m'
    fold_acc = []
    for i in range(cv):
        send_path = os.path.join(data_path,'{}CV'.format(cv),folds_string.format(i))
        modals = [modal for modal in modalities if modal.endswith(letter)]
        trainDS = dataset.train_ds(send_path, modals, **config['data_options']['train'])
        testDS = dataset.predict_ds(send_path, modals, **config['data_options']['test'])
        
        cvFold = str(i)
        save_models_here = os.path.join(save_path,'{}CV'.format(cv))        
        weights_filename = os.path.join(save_models_here,'Fold'+cvFold,network,method,'weights/')
        preds_filename = os.path.join(save_models_here,'Fold'+cvFold,network,method,'predictions/')
        testdata_filename = os.path.join(save_models_here,'Fold'+cvFold,network,method,'testdata/')
        os.makedirs(weights_filename, exist_ok=True)
        os.makedirs(preds_filename, exist_ok=True)
        os.makedirs(testdata_filename, exist_ok=True)
        tf.keras.backend.clear_session()

        n_trainsteps = folders.count_total(send_path,'train')//batch_size
        if network == 'cnn':
            model = vanillacnn.CNN()
        elif network == 'linearcombicnn':
            model = vanillacnn.linearCombiCNN()
        elif network == 'oldcnn':
            model = vanillacnn.oldCNN()
        elif network == 'vgg16':
            model = transfer_models.vgg16_net()
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
        
        tp, fp, tn, fn = metrics.perf_measure(y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
        acc = (tp+tn) / (tp+fp+tn+fn)
        print("test acc: ",acc)           
        fold_acc.append(acc)
        nf_aml,nf_cc = folders.count_fromFiles(send_path, 'test')
        if os.path.isdir(save_path) and os.path.exists(save_path):
            storepath = os.path.join(save_models_here,'Fold'+cvFold,network,method)
            os.makedirs(storepath, exist_ok=True)
        
        colnames = ['Network','Method','Modalities','Fold#','#AML(no)','#CCRCC(yes)','AUC','TP','FP','TN','FN','recall','specificity','f2','accuracy','avg_acc','stdev']
        
        eval_metrics = {k:0 for k in colnames}
        roundoff = 3
        eval_metrics['Network'] = network
        eval_metrics['Method'] = method
        eval_metrics['Modalities'] = ' '.join(modalities)
        eval_metrics['Fold#'] = i
        eval_metrics['#AML(no)'] = nf_aml
        eval_metrics['#CCRCC(yes)'] = nf_cc
        eval_metrics['TP'] = tp
        eval_metrics['FP'] = fp
        eval_metrics['TN'] = tn
        eval_metrics['FN'] = fn
        eval_metrics['accuracy'] = np.round_(acc,roundoff)
        eval_metrics['AUC'] = metrics.plot_roc(y_test.argmax(axis=-1),y_pred.argmax(axis=-1), storepath)
        metrics.plot_loss_acc(results, storepath, network=network)
        f2 = metrics.plot_confmat(tp, fp, tn, fn, storepath, roundoff)
        eval_metrics['f2'] = np.round_(f2,roundoff)
        eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
        eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)
        if i == cv - 1:
            eval_metrics['avg_acc'] = np.array(fold_acc).mean()
            eval_metrics['stdev'] = np.std(np.array(fold_acc))
        else:
            eval_metrics['avg_acc'] = 0.0
            eval_metrics['stdev'] = 0.0
        print(eval_metrics)

        metrics_path = os.path.join(save_path,'metrics_'+metrics_file_name+'.csv')
        if not os.path.exists(metrics_path):
            df = pd.DataFrame(columns=colnames)
            df = df.append(eval_metrics,ignore_index=True)
            #df.set_index('Network', inplace=True)
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

def meta_learner(whichos, config, max_steps, level0, level1, level0_filename):
    '''
    Train a model specified in level1 of config taking outputs from level0 networks already trained used train_stacked function
    Args:
        whichos: operation system linux/windows/remote
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
        level0 (str) : network at level0 (cnn|vgg16)
        level1 (str) : network at level1 (fc|xgb)
        level0_filename (str) : folder where level0 weights are stored
        '''
    
    config = load.load_config(config)
    cv = int(config['data_options']['cv'])
    fold_metrics = []
    for fold in range(cv):
        acc = foldwise_meta_learner(fold, whichos, config, max_steps, level0, level1, level0_filename)
        fold_metrics.append(acc)
        print("Fold: {}, Accuracy: {}".format(fold, acc))
    
    print("Average Accuracy: ", sum(fold_metrics)/len(fold_metrics))

    return


def foldwise_meta_learner(
    current_fold,
    whichos,
    config,
    max_steps,
    level_0,
    level_1,
    level0_filename,
):
    base_learning_rate = 0.01
    methods = ['CT', 'MRI']
    cv = config['data_options']['cv']
    save_models_here = os.path.join(config['data_options'][whichos]['save_path'],level0_filename,'{}CV'.format(cv),'Fold{}'.format(current_fold))      
    #save_models_here = config['data_options'][whichos]['save_models_here']
    
    #ytest will be same for CT or MRI (because subset of am_dc_ec_pc_tm)--loading from either one is enough
    testdataPath = os.path.join(save_models_here,level_0,methods[0],'testdata')
    Xtest = {}
    ytest = {}
    for method in methods:
        Xtest[method] = np.load(os.path.join(testdataPath,'X.npy'))
        ytest[method] = np.load(os.path.join(testdataPath,'y.npy'))
    #print("Xtest[CT].shape",Xtest['CT'].shape)
    #print("Xtest[MRI].shape",Xtest['MRI'].shape)
    #ytest = np.tile(np.squeeze(np.load(ytestPath)),[len(methods),1])#duplicating the labels of CT for MRI because labels are same
    
    #generate leave one out data for each method, for each class
    
    # models = load_models(save_models_here, level_0, base_learning_rate)
    # stacked_model = transfer_models.define_stacked_model(models)
    # stacked_dataset = stackData(Xtest, methods)
    return

def stackData(Xtest, methods):
    res = []
    assert (Xtest['CT'].shape[0] >= Xtest['MRI'].shape[0])
    res.append(Xtest['CT'][:Xtest['MRI'].shape[0],:,:,:])
    res.append(Xtest['MRI'])

    res = np.array(res)
    print(res.shape)
    return res

def fit_stacked_nn(pred_data, methods, ytest, max_steps, lr):
    stackedX, stackedytest = stacked_dataset(pred_data, ytest, methods)
    model = transfer_models.stackedNet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy'],
        )
    print("ytest.shape=",ytest.shape)
    print("ytest=",ytest)

    print("stackedytest.shape=",stackedytest.shape)
    print("stackedytest=",stackedytest)
    model.fit(stackedX, stackedytest, epochs=max_steps)
    return model

def fit_stacked_xgb(pred_data, methods, ytest, max_steps, lr):
    stackedX = stacked_dataset(pred_data, methods)
    model = transfer_models.gradientBoosting()
    new = np.argmax(ytest, axis=1)
    model.fit(stackedX, new)
    return model

def stacked_prediction(pred_data, methods, model):
    stackedX = stacked_dataset(pred_data, methods)
    yhat = model.predict(stackedX)
    return yhat


def load_models(save_models_here, level0_network, lr):
    base_learning_rate = lr
    both_models = {}
    methods = ['CT','MRI']
    for method in methods:
        if level0_network == 'cnn':
            model = vanillacnn.CNN(classifier_activation='softmax',num_classes=2)
        elif level0_network == 'vgg16':
            model = transfer_models.vgg16_net(classifier_activation='softmax', classifier_neurons=2)
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy'],
        )
        model_path = os.path.join(save_models_here,level0_network,method,'weights/')
        model.load_weights(model_path)
        both_models[method] = model

    return both_models