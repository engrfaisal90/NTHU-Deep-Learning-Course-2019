#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: faisalg
"""
import os
import numpy as np
import cv2
import imutils
import random

def preprocess(image):
    resized_image=cv2.resize(image,(128,128))
    if resized_image.shape[2]==1:
        resized_image = cv2.merge((resized_image,resized_image,resized_image))
    #cleaned_image = cv2.fastNlMeansDenoisingColored(resized_image)
    return resized_image

def OneHot(data):

    assert isinstance(data, np.ndarray)
    no_of_classes = np.max(data)+1
    out = np.zeros(shape=(len(data), no_of_classes))
    out[np.arange(len(data)), data] = 1
    return out
def augmentation(allimages,allabels):
    newImages=list(allimages)
    newlabels=list(allabels)
    for img,label in zip(allimages,allabels):
        ranomnum=random.randint(0,3)
        if ranomnum==0:
            img = np.fliplr(img)
        if ranomnum==1:
            img = imutils.rotate(img, 90)
        if ranomnum==2:
            img = imutils.rotate(img, 270)
        if ranomnum==3:
            img=imutils.rotate(img, 180)
        newImages.append(img)
        newlabels.append(label)
    return np.array(newImages),np.array(newlabels)
    


def read_images():
    print("Dataset reading and preprocessing started" )
    train_path = "data/train/"
    test_path="data/test/"
    train_images=[]
    train_labels=[]
    test_images=[]
    test_labels=[]
    all_folders=[]
    for folder in os.listdir(train_path):
        all_folders.append(folder)
    for label,folder in enumerate(all_folders):
        #read train images
        for image in os.listdir(train_path+folder):
            im=cv2.imread(train_path+folder+'/'+image)
            im=preprocess(im)
            train_images.append(im)
            train_labels.append(label)
        
        #read test images
        for image in os.listdir(test_path+folder):
            im=cv2.imread(test_path+folder+'/'+image)
            im=preprocess(im)
            test_images.append(im)
            test_labels.append(label)
    
    print("Data read and preprocessed" )
        
    classes=len(set(train_labels))
    return np.array(train_images),np.array(train_labels),np.array(test_images),np.array(test_labels),classes,all_folders
