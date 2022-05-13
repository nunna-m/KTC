'''
interface for training models
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
def train(
    whichos,
    modalities,
    method,
    network,
    config,
    max_steps,
    filename=None,
):
    '''
    Train a model with specified configs.
    Stores the cmdline args and other configs in one file for future ref, trains the model, stores results in csv and plots too.

    Args:
        whichos: operation system linux/windows/remote
        modalities (list[str]): the modalites being used
        method: CT or MRI or both (indicating used only ct modalities, or only mri modalities or use a combination of both)
        network: which network to use
        config (list[str]): config file paths (one or more) first one will be the main config and others will overwrite the main one or add to it
        max_steps (int): maximum training epochs
        filename: desired metrics filename, default is numAug_network_numberofepochs
    '''
    config = load.load_config(config)
    modalities = sorted(modalities, reverse=False)
    print("Modalities: %s"%modalities)
    print("Operating System: {}".format(whichos))
    print("Method: %s"%method)
    if not filename:
        metrics_file_name = '{}Aug_{}_{}eps'.format(len(config['data_options']['train']['aug_configs']),network,max_steps)
    else:
        metrics_file_name = filename
    save_path = os.path.join(config['data_options'][whichos]['save_path'],metrics_file_name,'_'.join(modalities))
    data_path = os.path.join(config['data_options'][whichos]['data_path'],'_'.join(modalities))
    cv = int(config['data_options']['cv'])
    batch_size = config['data_options']['train']['batch_size']
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
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), 
    ]

    fold_acc = []
    for i in range(cv):
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

        num_neurons = 2 #2 classifier neurons (binary classification)
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
        
        print('y_actual: {}'.format(y_test))
        print('y_pred: {}'.format(y_pred))
        
        tp, fp, tn, fn = metrics.perf_measure(y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
        acc = (tp+tn) / (tp+fp+tn+fn)
        print("test acc: ",acc)           
        fold_acc.append(acc)
        nf_aml,nf_cc = folders.count_fromFiles(send_path, 'test')
        if os.path.isdir(save_path) and os.path.exists(save_path):
            sendpath = os.path.join(save_models_here,'Fold'+cvFold)
            os.makedirs(sendpath, exist_ok=True)
        colnames = ['Network','Modalities','Fold#','#AML(no)','#CCRCC(yes)','AUC','TP','FP','TN','FN','recall','specificity','f2','accuracy','avg_acc']
        
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
        eval_metrics['AUC'] = metrics.plot_roc(y_test.argmax(axis=-1),y_pred.argmax(axis=-1), sendpath)
        metrics.plot_loss_acc(results, sendpath, network=network)
        f2 = metrics.plot_confmat(tp, fp, tn, fn, sendpath, roundoff)
        eval_metrics['f2'] = np.round_(f2,roundoff)
        eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
        eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)
        if i == cv - 1:
            eval_metrics['avg_acc'] = np.array(fold_acc).mean()
        else:
            eval_metrics['avg_acc'] = 0.0
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
    


