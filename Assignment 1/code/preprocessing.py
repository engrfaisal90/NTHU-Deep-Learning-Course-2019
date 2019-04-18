#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:36:34 2019

@author: faisalg
"""
import pandas as pd
import numpy as np





def preprocessing(data, train_test='train'):
   
    def one_hot_encoding(our_features):
        print("\n    started:   coversion of cateogorical colomns to one hot encoding")
        for i in our_features:
            if str(our_features[i].dtypes) == 'object':
                one_hot = pd.get_dummies(our_features[i], prefix = i ,drop_first=True)
                our_features = our_features.drop(i,axis = 1)
                our_features = our_features.join(one_hot)
        print("               shape of data is ", our_features.shape, "only categorical binary columns found")
        print("    finished:  coversion of cateogorical colomns to one hot encoding\n")
        return our_features
    
    def split_data(our_hot_encoded_features, train_samples_fraction):
        print("    started:   splitting into train test")
        train=our_hot_encoded_features.sample(frac=train_samples_fraction,random_state=200)
        test=our_hot_encoded_features.drop(train.index)
        print("               using 80% of data for training and 10% for testing")
        print("               Shape of train set =", train.shape)
        print("               Shape of train set =", test.shape)
        print("    finished:  splitting into train test\n")
        
        train_Y=train['G3']  
        train_X=train.drop ('G3',axis = 1)
        test_Y=test['G3']  
        test_X=test.drop ('G3',axis = 1)
        
        return train_X,train_Y,test_X, test_Y
    
    def normalize(data):
        for i in data:
            data[i]=(data[i]-data[i].mean())/data[i].std()
        return data
    
    
    
    
    print("started: pre-processing ")
    
    if (train_test=='train'):
        our_features=data[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','G3']]
        print("    Features extracted. our data has shape = ", our_features.shape)
        our_hot_encoded_features=one_hot_encoding(our_features)
        train_X,train_Y,test_X, test_Y= split_data(our_hot_encoded_features, 0.8)
        print("    started:   Normalizing ")
        train_X=normalize(train_X)
        test_X=normalize(test_X)
        print("    finished:  Normalaizing")
        print("finished: pre-processing\n")
        return np.array(train_X),np.array(train_Y),np.array(test_X), np.array(test_Y)
    else:
        our_features=data[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]
        print("    Features extracted. The shape of data to be predicted is = ", our_features.shape)
        our_hot_encoded_features=one_hot_encoding(our_features)
        print("    started:   Normalizing ")
        real_test_X=normalize(our_hot_encoded_features)
        print("    finished:  Normalaizing")
        print("finished: pre-processing\n")
        return real_test_X
    
    