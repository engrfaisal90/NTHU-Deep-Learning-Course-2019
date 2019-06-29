#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:02:36 2019

@author: faisalg
"""

import numpy as np


def split_test(data, fraction):
    print("\n\n#### Data Loaded  ####")
    train = data.sample(frac=fraction, random_state=200)
    test = data.drop(train.index)
    y_train = train['Activities_Types']
    x_train = train.drop('Activities_Types', axis=1)
    y_test = test['Activities_Types']
    x_test = test.drop('Activities_Types', axis = 1)
    y_train = np.subtract(y_train,1)
    y_test = np.subtract(y_test, 1)
    print("\n####  Train Test data Separated  ####")
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def making_batch(x, y, Size_of_batch=64, shuffle=False):
    assert len(x) == len(y)
    if shuffle:
        indices = np.random.permutation(len(x))
    for i in range(0, len(x) - Size_of_batch + 1, Size_of_batch):
        if shuffle:
            excerpt = indices[i:i + Size_of_batch]
        else:
            excerpt = slice(i, i + Size_of_batch)
        yield x[excerpt], y[excerpt]