#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import math
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import tensorflow as tf
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
from tensorflow.keras import layers,optimizers
from tensorflow import keras as lay
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve,average_precision_score


# In[ ]:


class All_Models:
    
    def weightvector(weightshape):
        weight=tf.Variable(tf.truncated_normal(weightshape, stddev=0.05))
        return weight

    def biasvector(bias_length):
        bias=tf.Variable(tf.constant(0.05, shape=[bias_length]))
        return bias

    def dropout(layer,prob):
        drop_out = tf.nn.dropout(layer, prob)
        return drop_out

    def dense_layer(inputsize,no_of_inputs,no_of_outputs,relu=True): 
        layer_weights = weightvector([no_of_inputs, no_of_outputs])
        layer_biases = biasvector(no_of_outputs)
        fc_layer = tf.matmul(inputsize, layer_weights) + layer_biases
        if relu==True:
            fc_layer = tf.nn.relu(fc_layer)
        return fc_layer,layer_weights

    def plotROCcurve(fpr, tpr, auc_roc):
        plt.figure(1)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        str1="AUC: %.3f" % auc_roc
        plt.legend(loc='upper left', labels=[ str1,'Random'])
        plt.show()
        
    def plotLoss(history):
        loss = history['loss']
        val_loss = history['val_loss']
        ep = range(1, len(loss) + 1)
        plt.plot(ep, loss, 'r', label='Training loss')
        plt.plot(ep, val_loss, 'g', label='Validation loss')
        plt.title('Loss Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plotAccu(history):
        accur = history['acc']
        val_accur = history['val_acc']
        ep = range(1, len(accur) + 1)
        plt.figure()
        plt.plot(ep, accur, 'r', label='Training acc')
        plt.plot(ep, val_accur, 'g', label='Validation acc')
        plt.title('Accuracy Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def LSTMModel():
        vocab_size = 10000
        model = lay.Sequential()
        model.add(layers.Embedding(vocab_size, 16))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation = 'sigmoid'))             
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model

    def plotPRCcurve(test_labels, pred,auc_prc):
        precision, recall, thresholds = precision_recall_curve(test_labels, pred)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PRC curve')
        str1="AUPRC: %.3f" % auc_prc
        plt.legend(loc='upper right', labels=[ str1])
        plt.show()

    def GRUModel():
        vocab_size = 10000
        model = lay.Sequential()
        model.add(layers.Embedding(vocab_size, 16))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation = 'sigmoid'))             
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model

    def RNNModel():
        vocab_size = 10000
        model = lay.Sequential()
        model.add(layers.Embedding(vocab_size, 16))
        model.add(layers.Dropout(0.2))
        model.add(layers.SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(1, activation = 'sigmoid'))             
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model


# In[ ]:


def preprocess(length):
    imdb = tf.keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=length)

    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=length)
    return train_data,train_labels,test_data,test_labels,word_index

def decode_review(text,reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def train_valid_set(train_data,train_labels):
    x_val = train_data[:10000]
    x_train = train_data[10000:]
    y_val = train_labels[:10000]
    y_train = train_labels[10000:]
    return x_train ,x_val,y_train,y_val


# In[ ]:


def question_1_c(x_train,x_val,y_train,y_val,test_data,test_labels):
    print("Making and Training Vanilla RNN")
    rnnModel = All_Models.RNNModel()
    rnnModel.summary()
    hist=rnnModel.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=[x_val, y_val])
    pred=rnnModel.predict(test_data)
    pred=pred>0.5
    history = hist.history
    All_Models.plotLoss(history)
    All_Models.plotAccu(history)
    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    auc_roc = auc(fpr, tpr)
    auc_prc = average_precision_score(test_labels, pred)
    All_Models.plotROCcurve(fpr, tpr, auc_roc)
    All_Models.plotPRCcurve(test_labels, pred,auc_prc)
    
    
def lstm (x_train,x_val,y_train,y_val,test_data,test_labels):
    print("Making and Training LSTM")
    lstmModel = All_Models.LSTMModel()
    lstmModel.summary()
    hist=lstmModel.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=[x_val, y_val])
    pred=lstmModel.predict(test_data)
    pred=pred>0.5
    history = hist.history
    All_Models.plotLoss(history)
    All_Models.plotAccu(history)
    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    auc_roc = auc(fpr, tpr)
    auc_prc = average_precision_score(test_labels, pred)
    All_Models.plotROCcurve(fpr, tpr, auc_roc)
    All_Models.plotPRCcurve(test_labels, pred,auc_prc)
    
def GRU(x_train,x_val,y_train,y_val,test_data,test_labels):
    print("Making and Training GRU")
    gruModel = All_Models.GRUModel()
    gruModel.summary()
    hist=gruModel.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=[x_val, y_val])
    pred=gruModel.predict(test_data)
    pred=pred>0.5
    history = hist.history
    All_Models.plotLoss(history)
    All_Models.plotAccu(history)
    fpr, tpr, thresholds = roc_curve(test_labels, pred)
    auc_roc = auc(fpr, tpr)
    auc_prc = average_precision_score(test_labels, pred)
    All_Models.plotROCcurve(fpr, tpr, auc_roc)
    All_Models.plotPRCcurve(test_labels, pred,auc_prc)
    
def question_1_f(x_train,x_val,y_train,y_val,test_data,test_labels):
    lstm (x_train,x_val,y_train,y_val,test_data,test_labels)
    GRU (x_train,x_val,y_train,y_val,test_data,test_labels)
    
def question_1_g():
    train_data,train_labels,test_data,test_labels,word_index=preprocess(length=256)
    x_train,x_val,y_train,y_val=train_valid_set(train_data,train_labels)
    question_1_c(x_train,x_val,y_train,y_val,test_data,test_labels)
    question_1_f(x_train,x_val,y_train,y_val,test_data,test_labels)
    
def main():
    train_data,train_labels,test_data,test_labels,word_index=preprocess(length=120)
    x_train,x_val,y_train,y_val=train_valid_set(train_data,train_labels)
    question_1_c(x_train,x_val,y_train,y_val,test_data,test_labels)
    question_1_f(x_train,x_val,y_train,y_val,test_data,test_labels)
    question_1_g()
    
if __name__ == "__main__":
    main()


# In[ ]:




