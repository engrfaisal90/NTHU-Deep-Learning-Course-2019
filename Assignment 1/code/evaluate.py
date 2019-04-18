#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:51:51 2019

@author: faisalg
"""
import numpy as np
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