#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:57:29 2019

@author: faisalg
"""

import layers as lay

def make_model(input_shape,n_layer):
    
    model = []
    neurons=int(input("Enter number of neuron for layer 1 : "))
    model.append(lay.Layer_D(input_shape,neurons))
    model.append(lay.rectified_linear())
    for i in range(1,n_layer):
        prev_neuron=neurons
        msg= "enter number of neuron for layer"+ str(i+1)+ " :  "
        neurons=int(input(msg))
        model.append(lay.Layer_D(prev_neuron,neurons))
        model.append(lay.rectified_linear())
    return model
