'''
interface to train stacked model
'''

from ktc.models.tf_models import transfer_models, vanillacnn
from ktc import dataset, folders
from ktc.utils import load, dump, metrics
import numpy as np
import pandas as pd
from tqdm.keras import TqdmCallback
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

folds_string = 'allSubjectPaths{}.yaml'


def train_stacked(
    whichos,
    method,
    level_0,
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

    print("Modalities: %s" % modalities)
    print("Operating System: {}".format(whichos))
    print("Method: %s" % method)
    cv = int(config['data_options']['cv'])
    batch_size = config['data_options']['train']['batch_size']
    if not filename:
        metrics_file_name = '{}_{}eps'.format(level_0, max_steps)
    else:
        metrics_file_name = filename
    save_path = os.path.join(
        config['data_options'][whichos]['save_path'], metrics_file_name)
    data_path = os.path.join(
        config['data_options'][whichos]['data_path'], '_'.join(modalities))
    dump.dump_options(
        os.path.join(save_path, 'options_'+level_0+'_{}CV.yaml'.format(cv)),
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

    if method == 'CT':
        letter = 'c'
    elif method == 'MRI':
        letter = 'm'
    fold_acc = []
    for i in range(cv):
        send_path = os.path.join(
            data_path, '{}CV'.format(cv), folds_string.format(i))
        modals = [modal for modal in modalities if modal.endswith(letter)]
        trainDS = dataset.train_ds(
            send_path, modals, **config['data_options']['train'])
        testDS = dataset.predict_ds(
            send_path, modals, **config['data_options']['test'])

        cvFold = str(i)
        save_models_here = os.path.join(save_path, '{}CV'.format(cv))
        weights_filename = os.path.join(
            save_models_here, 'Fold'+cvFold, f"{level_0}_{method}", 'weights/')
        preds_filename = os.path.join(
            save_models_here, 'Fold'+cvFold, f"{level_0}_{method}", 'predictions/')
        testdata_filename = os.path.join(
            save_models_here, 'Fold'+cvFold, f"{level_0}_{method}", 'testdata/')
        os.makedirs(weights_filename, exist_ok=True)
        os.makedirs(preds_filename, exist_ok=True)
        os.makedirs(testdata_filename, exist_ok=True)
        tf.keras.backend.clear_session()

        n_trainsteps = folders.count_total(send_path, 'train')//batch_size
        if level_0 == 'oldcnn':
            model = vanillacnn.oldCNN(classifier_activation='softmax')
        elif level_0 == 'vgg16_lastblocktrain':
            model = transfer_models.vgg16_net_lastblocktrain(
                classifier_activation='softmax')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=base_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=METRICS,
        )
        results = model.fit(
            trainDS,
            batch_size=batch_size,
            steps_per_epoch=n_trainsteps,
            epochs=max_steps,
            callbacks=[TqdmCallback(verbose=2)],
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
        np.save(testdata_filename+'X.npy', x_test)
        np.save(testdata_filename+'y.npy', y_test)

        y_pred = np.array(model.predict(testDS))
        np.save(preds_filename+'yhat.npy', y_pred)
        print('Saved into: %s' % weights_filename)

        tp, fp, tn, fn = metrics.perf_measure(
            y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
        acc = (tp+tn) / (tp+fp+tn+fn)
        print("test acc: ", acc)
        fold_acc.append(acc)
        nf_aml, nf_cc = folders.count_fromFiles(send_path, 'test')
        if os.path.isdir(save_path) and os.path.exists(save_path):
            storepath = os.path.join(
                save_models_here, 'Fold'+cvFold, f"{level_0}_{method}")
            os.makedirs(storepath, exist_ok=True)

        colnames = ['Level0_Network', 'Method', 'Modalities', 'Fold#',
                    '#AML(no)', '#CCRCC(yes)', 'AUC', 'TP', 'FP', 'TN', 'FN', 'recall', 'specificity', 'f2', 'accuracy', 'avg_acc', 'stdev']

        eval_metrics = {k: 0 for k in colnames}
        roundoff = 3
        eval_metrics['Level0_Network'] = level_0
        eval_metrics['Method'] = method
        eval_metrics['Modalities'] = ' '.join(modalities)
        eval_metrics['Fold#'] = i
        eval_metrics['#AML(no)'] = nf_aml
        eval_metrics['#CCRCC(yes)'] = nf_cc
        eval_metrics['TP'] = tp
        eval_metrics['FP'] = fp
        eval_metrics['TN'] = tn
        eval_metrics['FN'] = fn
        eval_metrics['accuracy'] = np.round_(acc, roundoff)
        eval_metrics['AUC'] = metrics.plot_roc(
            y_test.argmax(axis=-1), y_pred.argmax(axis=-1), storepath)
        metrics.plot_loss_acc(results, storepath, network=level_0)
        f2 = metrics.plot_confmat(tp, fp, tn, fn, storepath, roundoff)
        eval_metrics['f2'] = np.round_(f2, roundoff)
        eval_metrics['recall'] = np.round_((tp/(tp+fn)), roundoff)
        eval_metrics['specificity'] = np.round_((tn/(tn+fp)), roundoff)
        if i == cv - 1:
            eval_metrics['avg_acc'] = np.array(fold_acc).mean()
            eval_metrics['stdev'] = np.std(np.array(fold_acc))
        else:
            eval_metrics['avg_acc'] = 0.0
            eval_metrics['stdev'] = 0.0
        print(eval_metrics)

        metrics_path = os.path.join(
            save_path, 'metrics_'+metrics_file_name+'.csv')
        if not os.path.exists(metrics_path):
            df = pd.DataFrame(columns=colnames)
            df = df.append(eval_metrics, ignore_index=True)
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


def meta_learner(
        whichos,
        config,
        max_steps,
        level0,
        level1,
        level0_filename):
    '''
    Train a model specified in level1 of config taking outputs from level0 networks already trained using train_stacked function
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
        acc = foldwise_meta_learner(
            fold, whichos, config, max_steps, level0, level1, level0_filename)
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

    #current_fold = 0
    base_learning_rate = 0.01
    config = load.load_config(config)
    methods = ['CT', 'MRI']
    #print("Operating System: {}".format(whichos))
    #print("Methods: {}".format(methods))
    #level_0 = config['data_options']['network_info']['level_0']
    #level_1 = config['data_options']['network_info']['level_1']
    save_path = os.path.join(
        config['data_options'][whichos]['save_path'], level0_filename)
    save_models_here = os.path.join(save_path, '{}CV'.format(current_fold))
    # weights_filename = os.path.join(save_models_here,'Fold'+current_fold,f"{level_0}_{method}",'weights/')
    # preds_filename = os.path.join(save_models_here,'Fold'+current_fold,f"{level_0}_{method}",'predictions/')
    # testdata_filename = os.path.join(save_models_here,'Fold'+current_fold,f"{level_0}_{method}",'testdata/')

    # ytest will be same for CT or MRI (because subset of am_dc_ec_pc_tm)
    # loading from either one is enough
    test_data_path = os.path.join(
        save_models_here, 'Fold'+current_fold, f"{level_0}_{methods[0]}", 'testdata/y.npy')
    ytest_data = np.load(test_data_path)
    pred_data = dict()
    pred_data_counts = dict()
    for met in methods:
        pred_data_path = os.path.join(
            save_models_here, 'Fold'+current_fold, f"{level_0}_{met}", 'predictions/yhat.npy')
        pred_data[met] = np.load(pred_data_path)
        pred_data_counts[met] = pred_data[met].shape[0]
        #print(met, pred_data[met].shape)

    # make CT and MRI same shape
    if pred_data_counts['CT'] <= pred_data_counts['MRI']:
        pred_data['MRI'] = pred_data['MRI'][0:pred_data_counts['CT'], :]
        ytest_data = ytest_data[0:pred_data_counts['CT']]
        #print('new MRI shape:', pred_data['MRI'].shape)
    elif pred_data_counts['CT'] > pred_data_counts['MRI']:
        pred_data['CT'] = pred_data['CT'][0:pred_data_counts['MRI'], :]
        ytest_data = ytest_data[0:pred_data_counts['MRI']]
        #print('new CT shape:', pred_data['CT'].shape)

    #print(len(pred_data['CT']), len(pred_data['MRI']))
    if level_1 == 'fc':
        model = fit_stacked_model(pred_data=pred_data, methods=methods,
                                  ytest=ytest_data, max_steps=max_steps, lr=base_learning_rate)
    elif level_1 == 'xgb':
        model = fit_stacked_xgb(pred_data=pred_data, methods=methods,
                                ytest=ytest_data, max_steps=max_steps, lr=base_learning_rate)
    yhat = stacked_prediction(
        pred_data=pred_data, methods=methods, model=model)

    #print("Ytest data: {}".format(ytest_data))
    #print("Yhat data: {}".format(yhat))
    sv_ytest = np.argmax(ytest_data, axis=1)
    if level_1 == 'xgb':
        sv_yhat = yhat
    else:
        sv_yhat = np.argmax(yhat, axis=1)
    tp, fp, tn, fn = metrics.perf_measure(
        sv_ytest.argmax(axis=-1), sv_yhat.argmax(axis=-1))
    acc = (tp+tn) / (tp+fp+tn+fn)
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


def load_models(save_models_here, foldNum, level_0, lr):
    base_learning_rate = lr
    both_models = []
    met = ['CT', 'MRI']
    f = 0
    for i in range(len(met)):
        if level_0 == 'cnn':
            model = vanillacnn.CNN(
                classifier_activation='softmax', num_classes=2)
        elif level_0 == 'vgg16':
            model = transfer_models.vgg16_net(
                classifier_activation='softmax', classifier_neurons=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=base_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy'],
        )
        model_path = os.path.join(
            save_models_here, 'Fold'+foldNum, f"{level_0}_{met[i]}", 'weights/')
        model.load_weights(model_path)
        both_models.append(model)

    return both_models
