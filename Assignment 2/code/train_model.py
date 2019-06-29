#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:11:58 2019

@author: faisalg
"""

import evaluate_fucntions as evaf
import numpy as np


def forward(model, X):
    activations = []
    input = X

    for l in model:
        activations.append(l.forward(input))
        input = activations[-1]
    assert len(activations) == len(model)
    return activations

def predict(model,X):
    logits = forward(model,X)[-1]
    #print(logits)
    return logits.argmax(axis=-1)
def loss_function(model,X,y):
    activation = forward(model,X)
    #inputs = [X]+activation
    logits = activation[-1]
    loss = evaf.softmax_crossentropy(logits, y)
    return np.mean(loss)
    

def train(model,X,y):
    activation = forward(model,X)
    
    inputs = [X]+activation 
    zs = activation[-1]
    loss = evaf.softmax_crossentropy(zs,y)
    gradient_loss = evaf.grad_softmax_crossentropy(zs,y)
    for i in range(len(model))[::-1]:
        layer = model[i]
        gradient_loss = layer.backward(inputs[i],gradient_loss)
        
    return np.mean(loss)

