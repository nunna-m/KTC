import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt


def perf_measure(y_actual, y_hat):
    '''
    Calculate True positives, False Positives, True Negatives and False Negatives from predicted labels and actual labels
    Returns tuple of (TP, FP, TN, FN)
    Args:
        y_actual: actual labels (0 or 1 benign/cancerous)
        y_hat: predicted labels (0 or 1, major class is taken based on bigger number after softmax activation)
    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # print("y_actual: ",y_actual, len(y_actual))
    # print("y_hat: ",y_hat, len(y_hat))
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
    '''
    plots roc curve and returns auc
    Args:
        y_true: actual labels
        y_pred: predicted labels
        path: path to store the roc curve image
    '''
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
    '''
    Find the optimal probability cutoff point for a classification model related to event rate
    Can be used to find youden's index (currently not being used in this)
    Args:
        target : Matrix with dependent or target data, where rows are observations
        predicted : Matrix with predicted data, where rows are observations
    Returns:
        list type, with optimal cutoff value
    '''
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def fbeta(tp, fp, tn, fn, beta=2.0):
    '''return fbeta score based on tp, fp, tn, fn'''
    squared = pow(beta, 2)
    numerator = (1 + squared)*tp
    denominator = ((1 + squared)*tp) + squared*fn + fp
    return numerator/denominator

def plot_confmat(tp, fp, tn, fn, path, roundoff, beta=2.0):
    '''plot the confusion matrix and store the plot in path'''
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
    '''plot loss and accuracy curves (for now only training metrics are plotted because no validation data'''
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