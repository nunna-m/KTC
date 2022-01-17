import os
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
    for i in range(N_FOLDS):
        send_path = folds_path+str(i)+'.yaml'
        modals = [modal for modal in modalities if modal.endswith(letter)]
        train_ds = dataset.train_ds(send_path, modals, **config['data_options']['train'])
        test_ds = dataset.predict_ds(send_path, modals, **config['data_options']['test'])
        
        if level_0=='cnn':
            cvFold = str(i)
            weights_filename = os.path.join(save_models_here+cvFold,method,'weights/')
            preds_filename = os.path.join(save_models_here+cvFold,method,'predictions/')
            testdata_filename = os.path.join(save_models_here+cvFold,method,'testdata/')
            os.makedirs(weights_filename, exist_ok=True)
            os.makedirs(preds_filename, exist_ok=True)
            os.makedirs(testdata_filename, exist_ok=True)
            tf.keras.backend.clear_session()
            num_neurons = 2
            n_trainsteps = folders.count_samples(modals,data_path,'train')['total']//batch_size
            model = vanillacnn.CNN(classifier_activation='softmax',num_classes=num_neurons)
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
            print('Saved into: %s'%weights_filename)
            print("test loss, test acc: ",model.evaluate(test_ds))           

            x_test_numpy = []
            y_test_numpy = []
            for iter in test_ds.as_numpy_iterator():
                x_test_numpy.append(iter[0])
                y_test_numpy.append(iter[1])

            # x_test = np.concatenate([x_test_numpy[0],x_test_numpy[1]])
            # y_test = np.concatenate([y_test_numpy[0],y_test_numpy[1]])
            x_test = x_test_numpy[0]
            y_test = y_test_numpy[0]
            np.save(testdata_filename+'X.npy',x_test)            
            np.save(testdata_filename+'y.npy',y_test)
            numpy_ypred = model.predict(x_test)
            np.save(preds_filename+'yhat.npy',numpy_ypred)
            
            del model
            del results
    return  


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