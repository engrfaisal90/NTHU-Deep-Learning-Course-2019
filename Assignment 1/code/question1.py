#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import preprocessing as p
import regression as reg
import plot as plt
import pandas as pd

def ques_1():
    train_data = pd.read_csv("data/train.csv")
    print("Data Read")
    #preprocessing data to get train/test data and labels
    train_X,train_Y,test_X, test_Y= p.preprocessing(train_data,'train')
    #linear regression with no bias and regularization
    ypred_nobias_noreg,c_nobias_noreg=reg.LReg(train_X,train_Y,test_X, test_Y,lamb=0)
    print("The RMSE without with Simple LR is : ",c_nobias_noreg)
    #linear regression with regulriation
    ypred_reg,c_reg=reg.LReg(train_X,train_Y,test_X, test_Y,lamb=1)
    print("The RMSE of LR with regularization is: ",c_reg)
    #linear regression with regulriation and bias
    ypred_reg_bias,c_reg_bias=reg.LReg_with_bias_reg(train_X,train_Y,test_X, test_Y,lamb=1)
    print("The RMSE of LR with bias and reg is: ",c_reg_bias)
    #Bayesian regression with regulriation and bias
    ypred_bayesian,c_bayesian=reg.Bayesian_LR(train_X,train_Y,test_X, test_Y,alpha=1,lamb=1)
    print("The RMSE of Bayesian LR with bias and reg is: ",c_bayesian)
    
    #plot all the predictd values against ground truth
    plt.all_reg(test_Y,ypred_nobias_noreg,c_nobias_noreg,ypred_reg,c_reg,ypred_reg_bias,c_reg_bias,ypred_bayesian,c_bayesian)


