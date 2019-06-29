#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:00:09 2019

@author: faisalg
"""

import numpy as np



class Level:
    def forward(self, x):
        return x
    def backward(self, z, gradient_out):
        diag = np.eye(z.shape[1])
        return np.dot(gradient_out, diag)
    
    
    
class rectified_linear(Level):
 
    def forward(self, x):
        #print(np.maximum(0,x))
        return np.maximum(0,x)
    
    def backward(self, z, gradient_out):
        rectified_linear_grad = z > 0
        #print(rectified_linear_grad)
        return gradient_out*rectified_linear_grad

class layer_S(Level):

    def forward(self, z):
        exp_scores = np.exp(z)
        return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    


    


class Layer_D(Level):
    def __init__(self, inputs, outputs, lr=0.0001):
        self.lr = lr
        self.w = np.random.normal(loc=0.0, scale = np.sqrt(2/(inputs+outputs)), size = (inputs,outputs))
        self.b = np.zeros(outputs)
        
    def forward(self,x):
        d_p=np.dot(x,self.w) + self.b
        return d_p
    
    def adagrad(self,z,gradient_out):
        gti=np.zeros(np.shape(z))
        gti+=z**2
        adjusted_grad = gradient_out / (1e-6 + np.sqrt(gti))
        self.w = self.w - self.lr*adjusted_grad
    
    def Stoc_grad_d(self, z, gradient_out):
        g_w = np.dot(z.T, gradient_out)
        g_b = gradient_out.mean(axis=0)*z.shape[0]

        self.w= self.w - self.lr * g_w
        self.b= self.b - self.lr * g_b
    
    def backward(self,z,gradient_out):
        gradient_inp = np.dot(gradient_out, self.w.T)
        # which method of optimizer will be used
        self.Stoc_grad_d(z,gradient_out)
        
        return gradient_inp
    
    
    
    

