#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:02:21 2019

@author: faisalg
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score


def PCA_Visualization(dataset,y_values):
    coloumns = [ 'Feature'+str(i) for i in range(dataset.shape[1]) ]

    df = pd.DataFrame(dataset,columns=coloumns)

    df['label'] = y_values+1
    df['label'] = df['label'].apply(lambda i: str(i))

    pr_comp = PCA(n_components=2)
    pr_out = pr_comp.fit_transform(df[feat_cols].values)

    df['pca-one'] = pr_out[:,0]
    df['pca-two'] = pr_out[:,1] 

    colors=['b','g','r','c','m','y']
    classes=['dws','std','jog','ups','wlk','sit']
    plt.figure(figsize=[15,5])
    for i in range(len(df['pca-one'])):
        plt.scatter(df['pca-one'][i],df['pca-two'][i],c=colors[int(df['label'][i])-1])
    plt.title("Validation set visualization")
    plt.xlabel('pca-one')
    plt.ylabel('pca-two')
    plt.show()
    out_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = out_tsne.fit_transform(df[coloumns].values)

    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]

    plt.figure(figsize=[15,5])
    for i in range(len(df['pca-one'])):
        plt.scatter(df_tsne['y-tsne'][i],df_tsne['x-tsne'][i],c=colors[int(df['label'][i])-1])
    plt.title("TSNE visualization")
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')
    plt.show()

def softmax_crossentropy(logits,ydash):
    logits_ans = logits[np.arange(len(logits)),ydash]
    
    xentropy = - logits_ans + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy(logits,ydash):
    ones_ans = np.zeros_like(logits)
    ones_ans[np.arange(len(logits)),ydash] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_ans + softmax) / logits.shape[0]

def get_confusion_matrix(y_pred,y_real,threshold):
    y_pred=np.where(y_pred >= threshold, 1, 0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_real[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_real[i]!=y_pred[i]:
           FP += 1
        if y_real[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_real[i]!=y_pred[i]:
           FN += 1
   
    return [TP, FP, TN, FN]

    
def accuracy_precision(values):
    TP=values[0]
    FP=values[1]
    TN=values[2]
    FN=values[3]
    accuracy=(TP+TN)/(TP+FP+TN+FN)
    precision= TP/(TP+FP)
    return accuracy, precision

def accuracy_plot(train_acc,test_acc):
    plt.plot(train_acc,label='train accuracy: '+str(np.round(train_acc[-1],3)))
    plt.plot(test_acc,label='val accuracy: '+str(np.round(test_acc[-1],3)))
    plt.title("Model Accuracy ")
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()
    
def loss_plot(train_loss,test_loss):
    plt.plot(train_loss, label='Training loss: '+str(np.round(train_loss[-1],3)))
    plt.plot(test_loss, label='Validation Loss: '+str(np.round(test_loss[-1],3)))
    plt.title("Model Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(loc="lower left")
    plt.show()
    


def evaluat(y_true, pred):
    print("Accuracy score  : ", accuracy_score (y_true, pred))
    print("Precision Score (micro) : ",precision_score(y_true, pred, average='micro'))
    print("Precision Score (macro): ",precision_score(y_true, pred, average='macro'))
    print("F1 score (micro) : ",f1_score(y_true, pred, average='micro'))
    print("F1 score (macro) : ",f1_score(y_true, pred, average='macro'))
evaluate(y_val,pred)