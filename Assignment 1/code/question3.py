#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:09:10 2019

@author: faisalg
"""
import regression as reg
import preprocessing as p
import numpy as np
import pandas as pd




def ques_3():
    train_data = pd.read_csv("data/train.csv")
    print("Data Read")
    test_no_G3= pd.read_csv("data/test_no_G3.csv")
    #its the training data
    ID=np.array(train_data["ID"])
    train_X,train_Y,test_X, test_Y= p.preprocessing(train_data,'train')
    #label for this data has to be predicted 
    processed_test_x= p.preprocessing(test_no_G3,'test')



    #####question3(a)
    ypred_reg_bias=reg.LReg_with_bias_reg(train_X,train_Y,processed_test_x, test_Y=[],lamb=1,flag=1)
    print("Predicted G3 for test_no_G3 using linear regression is\n", list(np.round(ypred_reg_bias,2)))
    
    #converting data to required format and writing it in file
    data=""
    for i,j in zip(ID,ypred_reg_bias):
        data=data+str(i)+"\t"+str(j)+"\n"     
    file = open('output files/StudentID_1.txt','w') 
    file.write(data)
    file.close()
    print("data saved in file StudentID_1.txt\n\n")
    
  
    
    #####question 3 (b)
    #converting label of train data to binary
    train_Y =np.where(train_Y >= 10, 1, 0)
    #call Logistic_regression function .  This function uses logistic regression with bias and regularization
    y_pred_logis_reg=reg.Logistic_regression(train_X,train_Y,processed_test_x)
    #converting predicted value to binary based on threshod value 0.5
    y_pred_logis_reg=np.where(y_pred_logis_reg >= 0.48, 1, 0)
    print("Predicted G3 for test_no_G3 using logistic regression is\n", list(y_pred_logis_reg))
    
    #converting data to required format and writing it in file
    data=""
    for i,j in zip(ID,y_pred_logis_reg):
        data=data+str(i)+"\t"+str(j)+"\n"  
    file = open('output files/StudentID_2.txt','w') 
    file.write(data)
    file.close()
    print("data saved in file StudentID_1.txt\n\n")
