'''
CLI for train command
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
import tensorboard as tb
import numpy as np
# customs
from ktc.utils import get, store, load, dump
from ktc import dataset, folders
from ktc.models.tf_models import transfer_models, vanillacnn

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
    ds = dataset.train_ds(data_path, modalities, **config['data_options']['train'])
    if validate:
        #assert val_data_path is not None
        val_ds = dataset.eval_ds(data_path, modalities, **config['data_options']['eval'])
    else: val_ds = None

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

    model_initial = transfer_models.stacked_model()
    print(model_initial.summary())

    layer_dict = dict([(layer.name, layer) for layer in model_initial.layers])
    print(layer_dict)

    weights_path = '/home/maanvi/LAB/pre_trained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model_initial.load_weights(weights_path, by_name=True)

    for i, layer in enumerate(model_initial.layers):
        weigths = layer.get_weights()
        model_initial.layers[i].set_weights(weigths)
    
    model = keras.Model(model_initial.input, outputs=model_initial.output)

    features = []
    for i in tqdm(ds):
        image = np.expand_dims(i, axis=0)
        featurePredict = model.predict(image)
        features.append(featurePredict)
    featuresArr = np.array(features)
    featuresRes = np.reshape(featuresArr, (featuresArr.shape[0],
                                        featuresArr.shape[2],
                                        featuresArr.shape[3],
                                        featuresArr.shape[4]    ))

    featuresResXGB = np.reshape(featuresRes, (featuresRes.shape[0], featuresRes.shape[1]*featuresRes.shape[2]*featuresRes.shape[3]))






    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    batch_size = config['data_options']['train']['batch_size']
    n_trainsteps = folders.count_samples(modalities,data_path,'train')['total']//batch_size
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
        callbacks = [reduce_lr],
        verbose=1

    )
    
    #predict
    test_ds = dataset.predict_ds(data_path, modalities, **config['data_options']['test'])

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
        sendpath = os.path.join(save_path, 'graphs','_'.join(modalities))
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
    trainfile.plot_loss_acc(results, sendpath, metrics=METRICS)
    trainfile.plot_roc(y_numpy, y_pred, sendpath)
    f2 = trainfile.plot_confmat(tp, fp, tn, fn, sendpath, roundoff)
    eval_metrics['f2'] = np.round_(f2,roundoff)
    eval_metrics['recall'] = np.round_((tp/(tp+fn)),roundoff)
    eval_metrics['specificity'] = np.round_((tn/(tn+fp)),roundoff)

    print(eval_metrics)

    metrics_path = os.path.join(save_path,'graphs','metrics.csv')
    if not os.path.exists(metrics_path):
        df = pd.DataFrame(columns=colnames)
        df = df.append(eval_metrics,ignore_index=True)
        df.to_csv(os.path.join(save_path,'graphs','metrics.csv'))
    else:
        extra = pd.DataFrame([eval_metrics])
        extra.to_csv(os.path.join(save_path,'graphs','metrics.csv'), mode='a', header=False)
    
    return results