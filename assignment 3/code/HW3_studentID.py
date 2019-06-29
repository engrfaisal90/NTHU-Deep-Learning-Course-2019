#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import tensorflow as tf
import data_prepro as dp
import numpy as np
import math


def weightvector(weightshape):
    weight=tf.Variable(tf.truncated_normal(weightshape, stddev=0.05))
    return weight

def biasvector(bias_length):
    bias=tf.Variable(tf.constant(0.05, shape=[bias_length]))
    return bias

def layer_convolution(inputsize,no_of_inp_channels,size_of_filter,no_of_filters,pool=True): 
    layer_shape = [size_of_filter, size_of_filter, no_of_inp_channels, no_of_filters]
    layer_weights = weightvector(layer_shape)
    layer_biases = biasvector(no_of_filters)
    conv_layer = tf.nn.conv2d(inputsize,layer_weights,[1, 1, 1, 1],'SAME')
    conv_layer =conv_layer+ layer_biases
    if pool==True:
        conv_layer = tf.nn.max_pool(conv_layer,[1, 2, 2, 1],[1, 2, 2, 1],'SAME')
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer, layer_weights

def dropout(layer,prob):
    drop_out = tf.nn.dropout(layer, prob)
    return drop_out

def layer_flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def dense_layer(inputsize,no_of_inputs,no_of_outputs,relu=True): 
    layer_weights = weightvector([no_of_inputs, no_of_outputs])
    layer_biases = biasvector(no_of_outputs)
    fc_layer = tf.matmul(inputsize, layer_weights) + layer_biases
    if relu==True:
        fc_layer = tf.nn.relu(fc_layer)
    return fc_layer,layer_weights


def makelayer(regularization=False,classes=0):
    # check shape of data
    image_vector_array_size = X_train[0].shape[0] * X_train[0].shape[1] * X_train[0].shape[2]
    img_shape = X_train[0].shape[0]
    channels = X_train[0].shape[2]
    
    #unitilize placeholders
    inp = tf.placeholder(tf.float32, shape=[None, image_vector_array_size], name='inp')
    inp = tf.reshape(inp, [-1, img_shape, img_shape, channels])
    out_y = tf.placeholder(tf.float32, shape=[None, classes], name='out')
    y_true_cls = tf.argmax(out_y, axis=1)

    # make network
    con_lay1, con_wt1 = layer_convolution(inp,channels,size_of_filter=5,no_of_filters=16,pool=True)
    con_lay2, con_wt2 = layer_convolution(con_lay1,16,size_of_filter=5,no_of_filters=16,pool=True)
    flat_lay, output_flat = layer_flatten(con_lay2)
    den_lay1,den_wt1 = dense_layer(flat_lay,output_flat,256,True)
    #drop=dropout(layer,prob)
    den_lay2,den_wt2 = dense_layer(den_lay1,256,256,True)
    den_lay3,den_wt3 = dense_layer(den_lay2,256,classes,False)
    print(con_lay1,con_lay2,flat_lay,den_lay1,den_lay2,den_lay3,sep="\n")
    alpha=0.01
    y_pred = tf.nn.softmax(den_lay3)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=den_lay3,labels=out_y)
    if regularization==True:
        l2_loss=alpha*(tf.nn.l2_loss(con_wt1)+tf.nn.l2_loss(con_wt2)+tf.nn.l2_loss(den_wt1)+tf.nn.l2_loss(den_wt2)+tf.nn.l2_loss(den_wt3))
        cross_entropy=tf.add(l2_loss,cross_entropy)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    All_layers=[con_lay1,con_lay2,den_lay1,den_lay2,den_lay3]
    All_weights=[con_wt1,con_wt2,den_wt1,den_wt2,den_wt3]
    return optimizer,inp,out_y,cost,accuracy,All_layers,All_weights,y_pred
    


def Batch(inputs, targets, batchsize=64, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            
    inputs=inputs[excerpt]
    targets=targets[excerpt]
    yield inputs, targets

def TestBatch(inputs, targets, batchsize=128, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
    inputs=inputs[excerpt]
    targets=targets[excerpt]
    return inputs, targets


def optimize(X_train,y_train_onehot,X_val, y_val_onehot,num_iterations,train_batch_size = 128,regularization=False):
    optimizer,inp,out_y,cost,accuracy,All_layers,All_weights,y_pred=makelayer(regularization,classes)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    Historysaver=[ [],[],[], []]
    #saver = tf.train.Saver()
    for i in range(num_iterations):
        X_val, y_val_onehot=TestBatch(X_val, y_val_onehot, train_batch_size, shuffle=True)
        for x_batch, y_batch in Batch(X_train,y_train_onehot,train_batch_size,shuffle=True):
            session.run(optimizer, feed_dict={inp: x_batch,  out_y:y_batch})
        train_iter_loss,train_iter_acc = session.run([cost, accuracy], feed_dict={inp: x_batch,out_y: y_batch})
        test_iter_loss,test_iter_acc= session.run([cost, accuracy], feed_dict={inp: X_val,out_y:y_val_onehot})
        print("Accuracy : ", i, " Train : ",train_iter_acc, " Test : ",test_iter_acc )
        print("Cost : ", i, " Train : ",train_iter_loss, "Test : ",test_iter_loss )

        Historysaver[0].append(train_iter_acc)
        Historysaver[1].append(test_iter_acc)
        Historysaver[2].append(train_iter_loss)
        Historysaver[3].append(test_iter_loss)
    #model="\\mymodel"+str(regularization)
    #saver.save(session, model)
    return Historysaver,session,All_layers,All_weights,inp,y_pred



def weights_plot(layer_weight,session):
    weight=session.run(layer_weight)
    weight_vector=tf.reshape(weight,[-1])
    weight_vector=session.run(weight_vector)
    w=np.ones_like(weight_vector)/float(len(weight_vector))
    plt.figure()
    plt.hist(weight_vector, 100, weights=w)

def accuracy_plot(Historysaver):
    plt.figure()
    plt.plot(Historysaver[0][3:],label='train accuracy: '+str(np.round(Historysaver[0][-1],3)))
    plt.plot(Historysaver[1][3:],label='val accuracy: '+str(np.round(Historysaver[1][-1],3)))
    plt.title("Model Accuracy ")
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()
    
def loss_plot(Historysaver):
    plt.figure()
    plt.plot(Historysaver[2][3:], label='Training loss: '+str(np.round(Historysaver[2][-1],3)))
    plt.plot(Historysaver[3][3:], label='Validation Loss: '+str(np.round(Historysaver[3][-1],3)))
    plt.title("Model Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(loc="lower left")
    plt.show()
    
def plot_conv_layer(session,inp,All_folder,ypred,layer, image,clas):

    classification = session.run(tf.argmax(ypred, 1), feed_dict={inp: [image]})
    plt.imshow(image, cmap=plt.cm.binary)
    msg="original: "+ str(All_folder[clas])+  " predicted: "+ str(All_folder[classification[0]])
    plt.title(msg)
    plt.show()
    
    plt.figure()
    values = session.run(layer, feed_dict={inp: [image]})
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids,figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
def question1():
    print("Question 1")
    X_train,y_train, X_val, y_val,classes,all_folders= dp.read_images()
    y_train_onehot=dp.OneHot(y_train)
    y_val_onehot=dp.OneHot(y_val)
    classes = len(set(y_train))
    print("\n\n Data Shape : \n\nX_train shape",X_train.shape, "Y_train shape",y_train_onehot.shape,"X_validation shape", X_val.shape, "Y_validation shape",y_val_onehot.shape,sep="\n")
    print("\n\n\nQuestion 1 ended")
    return X_train,y_train_onehot,X_val,y_val_onehot,classes,all_folders
    
def question2(X_train,y_train_onehot,X_val, y_val_onehot,classes,regularization=False):
    
    print("\n Making model\n")
    print("\n Training network\n")
    Historysaver,session,All_layers,All_weights,inp,y_pred=optimize(X_train,y_train_onehot,X_val, y_val_onehot,num_iterations=500,train_batch_size = 64,regularization=regularization)
    print("\n Training network finished")
    return Historysaver,session,All_layers,All_weights,inp,y_pred

def question3(Historysaver,session,All_layers,All_weights):
    
    accuracy_plot(Historysaver)
    loss_plot(Historysaver)
    for weight in All_weights:
        weights_plot(weight,session)


def question4(All_layers,inp,image,All_folder,y_pred,clas):
    print("\n\nQuestion 4")
    for layer in range(2):
        plot_conv_layer(session,inp,All_folder,y_pred,layer=All_layers[layer], image=image,clas=clas)
    print("\n\nQuestion 4 ended")
        
def question5(X_train,y_train_onehot,X_val, y_val_onehot,classes):
    print("\n\nQuestion 5")
    Historysaver,session,All_layers,All_weights,inp,y_pred=question2(X_train,y_train_onehot,X_val, y_val_onehot,classes,regularization=True)
    question3(Historysaver,session,All_layers,All_weights)
    print("\n\nQuestion 5 ended")
    return Historysaver,session,All_layers,All_weights,inp,y_pred
    
def question6(X_train,y_train_onehot,X_val, y_val_onehot,classes):
    print("\n\nQuestion 6")
    print("\n Data Augmentation started")
    X_train,y_train_onehot=dp.augmentation(X_train,y_train_onehot)
    print("\n Data Augmentation Ended")
    Historysaver,session,All_layers,All_weights,inp,y_pred=question2(X_train,y_train_onehot,X_val, y_val_onehot,classes,regularization=False)
    question3(Historysaver,session,All_layers,All_weights)
    print("\n\nQuestion 6 ended")


X_train,y_train_onehot,X_val,y_val_onehot,classes,all_folders=question1()
print("\n\nQuestion 2")
Historysaver,session,All_layers,All_weights,inp,y_pred=question2(X_train,y_train_onehot,X_val, y_val_onehot,classes,regularization=False)
print("\n\nQuestion 2 ended")
print("\n\nQuestion 3")
question3(Historysaver,session,All_layers,All_weights)
print("\n\nQuestion 3 ended")
question4(All_layers,inp,X_val[1000],all_folders,y_pred,np.argmax(y_val_onehot[1000]))
question5(X_train,y_train_onehot,X_val, y_val_onehot,classes)
question6(X_train,y_train_onehot,X_val, y_val_onehot,classes)
question6(X_train,y_train_onehot,X_val, y_val_onehot,classes)


# In[ ]:




