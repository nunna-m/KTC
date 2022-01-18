from audioop import avg
import os
from pyexpat import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from tqdm.keras import TqdmCallback
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# customs
from ktc.utils import get, store, load, dump
from ktc import dataset, folders
from ktc.models.tf_models import transfer_models, vanillacnn

N_FOLDS = 5

def train_stacked(
    whichos,
    method,
    config,
    max_steps,
):
    '''
    Train a stacked model with the networks specified as level0 and level1 in the config file. Stores the cmdline args and other configs in one file for future ref, trains the model, stores results in csv and plots too.

    Args:
        whichos: operation system linux/windows/remote
        method: CT or MRI
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
        '''
    config = load.load_config(config)
    modalities = ['am', 'dc', 'ec', 'pc', 'tm']
    
    print("Modalities: %s"%modalities)
    print("Operating System: {}".format(whichos))
    print("Method: %s"%method)
    save_path = config['data_options'][whichos]['save_path']
    data_path = config['data_options'][whichos]['data_path']
    folds_path = config['data_options'][whichos]['folds_path']
    level_0 = config['data_options']['network_info']['level_0']
    level_1 = config['data_options']['network_info']['level_1']
    save_models_here = config['data_options'][whichos]['save_models_here']
    dump.dump_options(
        os.path.join(save_path, '_options_stackedModel_'+datetime.now().strftime("%m%d-%H%M")+'.yaml'),
        avoid_overwrite=True,
        config = config,
        save_path = save_path,
        data_path = data_path,
    )

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
    for i in range(N_FOLDS):
        send_path = folds_path+str(i)+'.yaml'
        modals = [modal for modal in modalities if modal.endswith(letter)]
        train_ds = dataset.train_ds(send_path, modals, **config['data_options']['train'])
        test_ds = dataset.predict_ds(send_path, modals, **config['data_options']['test'])
        
        cvFold = str(i)
        weights_filename = os.path.join(save_models_here+cvFold,level_0,method,'weights/')
        preds_filename = os.path.join(save_models_here+cvFold,level_0,method,'predictions/')
        testdata_filename = os.path.join(save_models_here+cvFold,level_0,method,'testdata/')
        os.makedirs(weights_filename, exist_ok=True)
        os.makedirs(preds_filename, exist_ok=True)
        os.makedirs(testdata_filename, exist_ok=True)
        tf.keras.backend.clear_session()
        num_neurons = 2
        n_trainsteps = folders.count_samples(modalities,data_path,'train')['total']//batch_size
        if level_0 == 'cnn':
            model = vanillacnn.CNN(classifier_activation='softmax',num_classes=num_neurons)
        elif level_0 == 'vgg16':
            model = transfer_models.vgg16_net(classifier_activation='softmax',classifier_neurons=num_neurons)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=METRICS,
        )
        results = model.fit(
            train_ds,
            batch_size = batch_size,
            steps_per_epoch=n_trainsteps,
            epochs=max_steps,
            callbacks = [TqdmCallback(verbose=2)],
            verbose=0,
        )
        model.save_weights(weights_filename)
        

        x_test_numpy = []
        y_test_numpy = []
        for iter in test_ds.as_numpy_iterator():
            x_test_numpy.append(iter[0])
            y_test_numpy.append(iter[1])

        # x_test = np.concatenate([x_test_numpy[0],x_test_numpy[1]])
        # y_test = np.concatenate([y_test_numpy[0],y_test_numpy[1]])
        x_test = x_test_numpy[0]
        y_test = y_test_numpy[0]

        #print(x_test_numpy, y_test_numpy)
        #print(len(x_test_numpy), len(y_test_numpy))
        print(len(x_test), len(y_test))
        np.save(testdata_filename+'X.npy',x_test)            
        np.save(testdata_filename+'y.npy',y_test)
        numpy_ypred = model.predict(x_test)
        np.save(preds_filename+'yhat.npy',numpy_ypred)
        loss, tp, fp, tn, fn, acc, precision, recall, AUC, prc = model.evaluate(test_ds)
        print('Saved into: %s'%weights_filename)
        print("test loss, test acc: ",loss, acc)           
        fold_acc.append(acc)
        del model
        del results
    avg_acc = np.array(fold_acc).mean()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("AVG ACC: {}".format(avg_acc))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    return  

def meta_learner(whichos, config, max_steps, level0, level1):
    '''
    Train a model specified in level1 of config taking outputs from networks specified as level0 in the config file.
    Args:
        whichos: operation system linux/windows/remote
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
        level0 (str) : network at level0 (cnn|vgg16)
        level1 (str) : network at level1 (fc|xgb)
        '''
    fold_metrics = []
    for fold in range(N_FOLDS):
        acc = foldwise_meta_learner(fold, whichos, config, max_steps, level0, level1)
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
):
    
    #current_fold = 0
    base_learning_rate = 0.01
    config = load.load_config(config)
    methods = ['CT', 'MRI']
    #print("Operating System: {}".format(whichos))
    #print("Methods: {}".format(methods))
    #level_0 = config['data_options']['network_info']['level_0']
    #level_1 = config['data_options']['network_info']['level_1']
    save_models_here = config['data_options'][whichos]['save_models_here']
    
    #ytest will be same for CT or MRI (because subset of am_dc_ec_pc_tm)
    #loading from either one is enough
    test_data_path = os.path.join(save_models_here+str(current_fold),level_0,methods[0],'testdata/y.npy')
    ytest_data = np.load(test_data_path)
    pred_data = dict()
    pred_data_counts = dict()
    for met in methods:
        pred_data_path = os.path.join(save_models_here+str(current_fold),level_0,met,'predictions/yhat.npy')
        pred_data[met] = np.load(pred_data_path)
        pred_data_counts[met] = pred_data[met].shape[0]
        #print(met, pred_data[met].shape)
    
    #make CT and MRI same shape
    if pred_data_counts['CT'] <= pred_data_counts['MRI']:
        pred_data['MRI'] = pred_data['MRI'][0:pred_data_counts['CT'],:]
        ytest_data = ytest_data[0:pred_data_counts['CT']]
        #print('new MRI shape:', pred_data['MRI'].shape)
    elif pred_data_counts['CT'] > pred_data_counts['MRI']:
        pred_data['CT'] = pred_data['CT'][0:pred_data_counts['MRI'],:]
        ytest_data = ytest_data[0:pred_data_counts['MRI']]
        #print('new CT shape:', pred_data['CT'].shape)
    
    #print(len(pred_data['CT']), len(pred_data['MRI']))
    if level_1 == 'fc':
        model = fit_stacked_model(pred_data=pred_data, methods=methods, ytest=ytest_data, max_steps=max_steps, lr=base_learning_rate)
    elif level_1 == 'xgb':
        model = fit_stacked_xgb(pred_data=pred_data, methods=methods, ytest=ytest_data, max_steps=max_steps, lr=base_learning_rate)
    yhat = stacked_prediction(pred_data=pred_data, methods=methods, model=model)

    #print("Ytest data: {}".format(ytest_data))
    #print("Yhat data: {}".format(yhat))
    sv_ytest = np.argmax(ytest_data, axis=1)
    if level_1=='xgb':
        sv_yhat = yhat
    else:
        sv_yhat = np.argmax(yhat, axis=1)
    acc = accuracy_score(sv_ytest, sv_yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
    return acc

def stacked_dataset(pred_data, methods):
    stackX = None

    for met in methods:
        yhat = pred_data[met]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    print("Stacked shape: {}".format(stackX.shape))
    return stackX

def fit_stacked_model(pred_data, methods, ytest, max_steps, lr):
    stackedX = stacked_dataset(pred_data, methods)
    model = transfer_models.stackedNet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy'],
        )
    model.fit(stackedX, ytest, epochs=max_steps)
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
    both_models = []
    met = ['CT','MRI']
    f = 0
    for i in range(len(met)):
        if level0_network == 'cnn':
            model = vanillacnn.CNN(classifier_activation='softmax',num_classes=2)
        elif level0_network == 'vgg16':
            model = transfer_models.vgg16_net(classifier_activation='softmax', classifier_neurons=2)
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy'],
        )
        model_path = os.path.join(save_models_here+str(f),level0_network,met[i],'weights/')
        model.load_weights(model_path)
        both_models.append(model)

    return both_models

# for i in range(N_FOLDS):
#         send_path = folds_path+str(i)+'.yaml'
#         #CT data loading
#         ct_modalities = [modal for modal in modalities if modal.endswith('c')]
#         ct_train_ds = dataset.train_ds(send_path, ct_modalities, **config['data_options']['train'])
#         ct_test_ds = dataset.predict_ds(send_path, ct_modalities, **config['data_options']['test'])
#         #MRI data loading
#         mri_modalities = [modal for modal in modalities if modal.endswith('m')]
#         mri_train_ds = dataset.train_ds(send_path, mri_modalities, **config['data_options']['train'])
#         mri_test_ds = dataset.predict_ds(send_path, mri_modalities, **config['data_options']['test'])
        
#         if level_0=='cnn':
#             #CT training
#             method = 'CT'
#             cvFold = str(i)
#             weights_filename = os.path.join(save_models_here+cvFold,method,'weights/')
#             preds_filename = os.path.join(save_models_here+cvFold,method,'predictions/')
#             testdata_filename = os.path.join(save_models_here+cvFold,method,'testdata/')
#             os.makedirs(weights_filename, exist_ok=True)
#             os.makedirs(preds_filename, exist_ok=True)
#             os.makedirs(testdata_filename, exist_ok=True)
#             tf.keras.backend.clear_session()
#             num_neurons = 2
#             n_trainsteps = folders.count_samples(ct_modalities,data_path,'train')['total']//batch_size
#             model = vanillacnn.CNN(classifier_activation='softmax',num_classes=num_neurons)
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#                 loss=tf.keras.losses.CategoricalCrossentropy(),
#                 metrics=METRICS,
#             )
#             results = model.fit(
#                 ct_train_ds,
#                 batch_size = batch_size,
#                 steps_per_epoch=n_trainsteps,
#                 epochs=max_steps,
#                 callbacks = [TqdmCallback(verbose=2)],
#                 verbose=0,
#             )
#             model.save_weights(weights_filename)
#             print('Saved into: %s'%weights_filename)
#             print("test loss, test acc: ",model.evaluate(ct_test_ds))           

#             x_test_numpy = []
#             y_test_numpy = []
#             for iter in ct_test_ds.as_numpy_iterator():
#                 x_test_numpy.append(iter[0])
#                 y_test_numpy.append(iter[1])

#             x_test = np.concatenate([x_test_numpy[0],x_test_numpy[1]])
#             y_test = np.concatenate([y_test_numpy[0],y_test_numpy[1]])
#             np.save(testdata_filename+'X.npy',x_test)            
#             np.save(testdata_filename+'y.npy',y_test)
#             numpy_ypred = model.predict(x_test)
#             np.save(preds_filename+'yhat.npy',numpy_ypred)
            
#             del model
#             del results
            
#             #MRI training
#             method = 'MRI'
#             cvFold = str(i)
#             weights_filename = os.path.join(save_models_here+cvFold,method,'weights/')
#             preds_filename = os.path.join(save_models_here+cvFold,method,'predictions/')
#             testdata_filename = os.path.join(save_models_here+cvFold,method,'testdata/')
#             os.makedirs(weights_filename, exist_ok=True)
#             os.makedirs(preds_filename, exist_ok=True)
#             os.makedirs(testdata_filename, exist_ok=True)
#             tf.keras.backend.clear_session()
#             num_neurons = 2
#             n_trainsteps = folders.count_samples(mri_modalities,data_path,'train')['total']//batch_size
#             model = vanillacnn.CNN(classifier_activation='softmax',num_classes=num_neurons)
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#                 loss=tf.keras.losses.CategoricalCrossentropy(),
#                 metrics=METRICS,
#             )
#             results = model.fit(
#                 mri_train_ds,
#                 batch_size = batch_size,
#                 steps_per_epoch=n_trainsteps,
#                 epochs=max_steps,
#                 callbacks = [TqdmCallback(verbose=2)],
#                 verbose=0,
#             )
#             model.save_weights(weights_filename)
#             print('Saved into: %s'%weights_filename)
#             print("test loss, test acc: ",model.evaluate(mri_test_ds))           

#             x_test_numpy = []
#             y_test_numpy = []
#             for iter in mri_test_ds.as_numpy_iterator():
#                 x_test_numpy.append(iter[0])
#                 y_test_numpy.append(iter[1])

#             x_test = np.concatenate([x_test_numpy[0],x_test_numpy[1]])
#             y_test = np.concatenate([y_test_numpy[0],y_test_numpy[1]])
#             np.save(testdata_filename+'X.npy',x_test)            
#             np.save(testdata_filename+'y.npy',y_test)
#             numpy_ypred = model.predict(x_test)
#             np.save(preds_filename+'yhat.npy',numpy_ypred)
            
#             del model
#             del results



# tf.keras.backend.clear_session()
# new_model = vanillacnn.CNN(classifier_activation='softmax',num_classes=2)
# new_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['categorical_accuracy'],
# )
# new_model.fit(ct_test_ds, batch_size=1, epochs=1)
# new_model.load_weights(filename)
# acc = new_model.evaluate(ct_test_ds, verbose=2)
# print("Restored model, accuracy:",acc)