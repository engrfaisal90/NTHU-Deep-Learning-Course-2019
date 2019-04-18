#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:25:11 2019

@author: faisalg
"""

import pandas as pd
import numpy as np

def LReg(x,y,test_X, test_Y,lamb=0):
    #make a regulizer term lambda * identity matrix if lambda=0 reg=0 so no effect on weights. 
    reg=lamb*np.identity(np.array(x).shape[1])
    #calculate weights
    weights = np.linalg.inv(np.add((np.transpose (x) @ x), reg)) @ (np.transpose(x) @ y) 
    #predict the test set
    y_pred= test_X @ weights
    #calculate RMSE with regulizer or without regulizer
    if (lamb==0):
        co=np.sqrt(((y_pred - test_Y) ** 2).mean())
    else:
        co=np.sqrt(((y_pred - test_Y) ** 2).mean() + ((np.transpose(weights) @ weights)/2))
 
    return y_pred,co
    
    
def LReg_with_bias_reg(x=[],y=[],test_X=[], test_Y=[],lamb=1,flag=0):
    #add coloumns of 1 at start for bias value
    x=np.insert(x, 0, 1, axis=1)
    #make identity matrix of size features in x  and make its [0,0]=0 because we dont want to add penalty to bias term
    ident=np.identity(np.array(x).shape[1])
    ident[0,0]=0
    reg=lamb*ident
    weights = np.linalg.inv(np.add((np.transpose (x) @ x), reg)) @ (np.transpose(x) @ y)
    bias=weights[0]
    weights=weights[1:]
    y_pred= (test_X @ weights)+bias
    #flag is set to for question 3. we dont have ground truths and only we need to predict and dont need to calculate MSE
    if flag==1:
        return y_pred
    else:
        co=np.sqrt(((y_pred - test_Y) ** 2).mean() + ((np.transpose(weights) @ weights)/2))
        return y_pred,co

    

def Bayesian_LR(x,y,test_X, test_Y,alpha=1,lamb=0):
    #add coloumns of 1 at start for bias value 
    x=np.insert(x, 0, 1, axis=1)
    #create column of initial meo having values 0
    meo=np.zeros(((x.shape)[1]))
    #calculate initial delta value
    delta=(1/alpha)*np.identity(18)
    # update delta and meo value
    delta_m=np.linalg.inv((np.transpose(x) @ x)+ np.linalg.inv(delta))
    meo_m=delta_m @ (np.add((np.transpose(x) @ y) , np.linalg.inv(delta) @ meo ))
    #use new meo as weights (mean of posterior)
    y_pred= (test_X @ meo_m[1:])+meo_m[:1]
    co=np.sqrt(((y_pred - test_Y) ** 2).mean()  + ((np.transpose(meo_m[1:]) @ meo_m[1:])/2))
    return y_pred,co




def gradientDescent(x, y, lear_rate, num_Iterations):
    m,n=np.shape(x)
    weights = np.ones(n)
    xTrans = x.transpose()
    for i in range(0, num_Iterations):
        
        hyp = 1.0 / (1.0 + np.exp(-(x @ weights)))
        loss = hyp - y
        #cost = (-1)*np.sum((y*np.log(hyp)) + ((1-y)*np.log(1-hyp))) / (m)
        gradient = (xTrans @ loss)  / m
        weights = weights - lear_rate * gradient
    return weights

def Logistic_regression(x,y,test_X):
    numIterations= 10000
    lear_rate = 0.01
    x=np.insert(x, 0, 1, axis=1) 
    ident=np.identity(np.array(x).shape[1])
    ident[0,0]=0
    #reg=lamb*ident
    weights = gradientDescent(x, y, lear_rate,numIterations)
    y_pred =  test_X @ weights[1:] + weights[:1]
    y_pred = 1.0 / (1.0 + np.exp((-1) *y_pred))
    return y_pred  
    
  