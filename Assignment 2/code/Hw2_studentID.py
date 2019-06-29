#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:55:14 2019

@author: faisalg
"""
import pandas as pd
import preprocessing as p
import create_model as M
import train_model as training
import evaluate_fucntions as evaf
import numpy as np
import sys

data = pd.read_csv('Data.csv')
x_train, y_train, x_test, y_test = p.split_test(data,0.8)

print("\n\n####  Creating Model  #####")
print("\n###  Model Architecture Making #####")

n_layer=int(input("Enter number of layers : ")) 
model=M.make_model(x_train.shape[1],n_layer)


Size_of_batch=int(input("Enter batch size : "))
n_epoches = int(input("Enter number of epoches : "))

def ques1_part1(x_train, y_train, x_test, y_test,Historysaver):
    for epoch in range(n_epoches):
        print(" Epoch ", epoch,"/", n_epoches )
        
        for x_batch,y_batch in p.making_batch(x_train,y_train,Size_of_batch,shuffle=True):
            sys.stdout.write("-")
            loss=training.train(model,x_batch,y_batch)
            sys.stdout.flush()
        Historysaver[0].append(np.mean(training.predict(model,x_train)==y_train))
        Historysaver[1].append(np.mean(training.predict(model,x_test)==y_test))
        Historysaver[2].append(training.loss_function(model,x_test, y_test))
        Historysaver[3].append(loss)    
        
    evaf.accuracy_plot(Historysaver)
    evaf.loss_plot(Historysaver)
    
    
def ques1_part2(x_test,y_test):
    pred= training.predict(model,x_test)
    print(pred,y_test)
    evaf.evaluate(y_test,pred)
    
def ques1_part3(x_test, y_test ):
    evaf.PCA_Visualization(x_test,y_test)

def ques_2(data):
    test_data= pd.read_csv("Test_no_Ac.csv")
    pred= training.predict(model,test_data.values)
    for i,j in enumerate(pred):
        data=data+str(i)+"\t"+str(j)+"\n"
    check=int(input("Do you want to overwrite the file created? press 1 for yes"))
    if check==1:
        file = open('106062864_answer.txt','w') 
        file.write(data)
        file.close()
        print("Data saved in file 106062864_answer.txt\n\n")




Historysaver=[ [],[],[], []]

print("############  Question 1 part 1   ####################")
ques1_part1(x_train, y_train, x_test, y_test,Historysaver)
print("\n\n####################################################")
print("###################################################")
print("############  Question 1 part 2   ####################")
ques1_part2(x_test,y_test)
print("\n\n####################################################")
print("###################################################")
print("############  Question 1 part 4   #####################")
ques1_part3(x_test, y_test )

print("\n\n####################################################")
print("###################################################")
print("############  Question 2   #####################")
ques_2(data)











 