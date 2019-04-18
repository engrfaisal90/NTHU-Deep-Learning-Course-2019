#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:28:58 2019

@author: faisalg
"""

import numpy as np
import preprocessing as p
import regression as reg
import plot as myplot
import evaluate as ev
import pandas as pd

    

    

def ques_2():
    train_data = pd.read_csv("data/train.csv")
    print("Data Read")
    #convert G3 coloumn to binary
    train_data['G3'] =np.where(train_data['G3'] >= 10, 1, 0)
    #split to train test set
    train_X,train_Y,test_X, test_Y= p.preprocessing(train_data,'train')
    
    
    
    #call LReg_with_bias_reg function .  it calculates linear regression with bias and regularization
    ypred_reg_bias,c_reg_bias=reg.LReg_with_bias_reg(train_X,train_Y,test_X, test_Y,lamb=1)
    
    #get confusion matrx using three diffrent thresold values for predicted G#
    conf_matrix_1=ev.get_confusion_matrix(ypred_reg_bias,test_Y,0.1)
    conf_matrix_5=ev.get_confusion_matrix(ypred_reg_bias,test_Y,0.5)
    conf_matrix_9=ev.get_confusion_matrix(ypred_reg_bias,test_Y,0.9)
    
    #Calculate Accuracy and precision values and plot confusion matrix for all thresold values
    accuracy, precision= ev.accuracy_precision(conf_matrix_1)
    myplot.plot_confusion_matrix(conf_matrix_1,accuracy,np.round(precision,4), 0.1)
    accuracy, precision= ev.accuracy_precision(conf_matrix_5)
    myplot.plot_confusion_matrix(conf_matrix_5,accuracy,np.round(precision,4), 0.5)
    accuracy, precision= ev.accuracy_precision(conf_matrix_9)
    myplot.plot_confusion_matrix(conf_matrix_9,accuracy,np.round(precision,4), 0.9)
    
    
    
    
    #call Logistic_regression function .  This function uses logistic regression with bias and regularization
    y_pred_logis_reg=reg.Logistic_regression(train_X,train_Y,test_X)
    #get confusion matrx using three diffrent thresold values for predicted G#
    conf_matrix_1=ev.get_confusion_matrix(y_pred_logis_reg,test_Y,0.1)
    conf_matrix_5=ev.get_confusion_matrix(y_pred_logis_reg,test_Y,0.5)
    conf_matrix_9=ev.get_confusion_matrix(y_pred_logis_reg,test_Y,0.9)
    
    
    #Calculate Accuracy and precision values and plot confusion matrix for all thresold values
    accuracy, precision= ev.accuracy_precision(conf_matrix_1)
    myplot.plot_confusion_matrix(conf_matrix_1,accuracy,np.round(precision,4), 0.1)
    accuracy, precision= ev.accuracy_precision(conf_matrix_5)
    myplot.plot_confusion_matrix(conf_matrix_5,accuracy,np.round(precision,4), 0.5)
    accuracy, precision= ev.accuracy_precision(conf_matrix_9)
    myplot.plot_confusion_matrix(conf_matrix_9,accuracy,np.round(precision,4), 0.9)
    
    print("All the graphs are printed after question 3")
