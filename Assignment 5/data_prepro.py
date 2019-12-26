#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:24:37 2019

@author: faisalg
"""

import numpy as np

def input_data():
    from emnist import extract_training_samples,extract_test_samples
    train_images, train_labels = extract_training_samples('balanced')
    test_images, test_labels = extract_test_samples('balanced')
    
    tra_images=[]
    tra_labels=[]
    tes_images=[]
    tes_labels=[]
    for i in range (len(train_images)):
        if train_labels[i]>9 and train_labels[i]<36:
            tra_labels.append(train_labels[i])
            tra_images.append(train_images[i])
    for i in range (len(test_images)):
        if test_labels[i]>9 and test_labels[i]<36:
            tes_labels.append(test_labels[i])
            tes_images.append(test_images[i])
    tra_images=np.array(tra_images)
    tes_images=np.array(tes_images)
    
    tra_images = tra_images.astype('float32') / 255.0
    tes_images = tes_images.astype('float32') / 255.0
    return tra_images, tra_labels, tes_images, tes_labels