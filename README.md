# Deep Learning from scratch (No pytortch and tensorflow)-2019
This repository contain all the assignments of Graduate level course (deep learning of spring semester 2019 at NTHU.

Each assignment contain "Assignment.pdf" file , "Student_ID.pdf" file and code folder.</br>
  Assignment.pdf file contain questions of the assignment.</br>
  Student_ID.pdf is the report file and contain answers to the question and detail about the code.</br>
  Code folder contain all the code to solve those questions. the code folder might additionaly contain "data" folder and "output files" folder.</br>
    "Data" folder contain all the related used in the assignment.</br>
    "output file" contain the files generate in the assignment. </br>


## Assignmnet 1 code</br>
Although detail is included in the Assignment.pdf" file and "Student_ID.pdf" in assignment folder. Here is outline of the code.</br>
  * hw1 StudentID.py contain the main function.</br>
  * Question1.py. gives detail implemtaion of question 1 of the assignment.</br>
  * Question2.py. gives detail implementaion of question 2 of the assignment.</br>
  * Question3.py. gives detail implementaion of question 3 of the assignment.</br>
  * Preprocessing.py preprocess the data. it contains following functions.</br>
    + One-hot-encoding ~ convert the features to one hot encoding.</br>
    + split-data ~ split the data into train and test data and also separate the label.</br>
    + normalize ~ it normalizes each feature according to requirement.</br>
  * Regression.py includes all the functions required in regression.</br>
    + Lreg ~ Implement a linear regression model without the bias term to predict G3.   if lambda =0 is passed. </br>
      Also it Implement a regularized linear regression model without the bias.  if lambda value is passed.</br>
    + LReg_with_bias_reg ~ Implement a regularized linear regression model without the bias. </br>
    + Bayesian_LR ~ Implement a Bayesian linear regression model with the bias term. </br>
    + Logistic_regression ~ Implement a Logistic_regression model.</br>
    + Gradient_descent ~ Implements a function that calculates gradient desscent.</br>
  * Evaluate.py includes the function that evaluates the algorithms.</br>
    + get_confusion_matrix calculates True positive, True negative, False positive and false negative values.</br>
    + Accuracy_precision calulayes Accuracy and precision. </br>
  * Plot.py plots all the required figures in the assignment. </br>
    + all_reg ~ plots ground truth values against all models. </br>
    + plot_confusion_matrix plot the confusion matrix. </br>
    
## Assignmnet 2 code</br>
  The details are included in the Assignment.pdf" file and "Student_ID.pdf" in assignment folder. Here is outline of the code.</br>
  * hw2 StudentID.py contain the functions, which solves the realated questions in the assignmnet.</br>
  * Preprocessing.py preprocess the data. it contains following functions.</br>
    + split-test ~ split the data into train and test data and also separate the label.</br>
    + making_batch ~ creates batch data.</br>
  * Evaluate.py includes the function that evaluates the algorithms.</br>
    + get_confusion_matrix ~ calculates True positive, True negative, False positive and false negative values.</br>
    + Accuracy_precision ~ calulates Accuracy and precision. </br>
    + accuracy_plot ~ plots the accuracy curve
    + loss_plot ~ plots the loss curve
    + evaluat ~ calculates and prints the macro and micro precision and F1-Score.
    + PCA_Visualization ~ project the validation set onto a 2D space by t-SNE
  * layers.py ~ the classes contain functions through which diffrent layers are created. </br>
    + rectified_linear ~ create a RELU layer. </br>
    + layer_S ~ create a softmax layer.
    + Layer_D ~ create a dense layer.
      + The forward function are responsible for forward pass.
      + the backward function are resposible for backward pass.
  * create_model.py ~ Creates the model.
  * train_model.py ~ trains the created model.
    
## Assignmnet 3 code</br>
  The details are included in the Assignment.pdf" file and "Student_ID.pdf" in assignment folder. Here is outline of the code.</br>
* Read_images: this function read all the data.
* Onehot: this function converts the labels into one hot vectors.
* Preprocess: this function resize, rescale, check channels and remove noise if necessary. Augmentation: this function augments the data by flipping all the images.
* Weight_vector: this function generates weight tensor of a shape required for a layer. Bias_vector: this function generates Bias tensor of a shape required for a layer. Layer_convolution creates a convolutional layer, number of inputs, channels, size of filters, number of filters and to use pooling or not is required for tgis function.
* Layer_flatten: flattens the output from convolutional layer
* Dense layer: create a dense layer of input and output connection should be passed. Makelayer: creates the whole Network architecture
* Batch: creates batch training data
* TestBatch: randomly takes a batch of test data for validation.
* Optimize: Optimize function is heart of the program which trains the network, Data, number of iterations and batch size and whether regularization should be done or is to be passed to this function.
* Weights_plot: It plots the histograms of the weights of a layer.
* Accuracy_plot: It plots the training and validation accuracy, calculated in every epoch. Loss_plot: It plots the training and validation loss, calculated in every epoch. Plot_conv_layer: This function plots outputs of the convolution layers.
* Question1: read and preprocess the data
* Question2: it trains the Network
* Question3: plots the accuracy, loss and weights in layers.
* Question4: it plots all the images.
* Question5: it trains the network with Regularization
* Question6: it applies data augmentation and then train a network.


## Assignmnet 3,4,5 code</br>
The detail of all other assignments can be found in the respective folders.
