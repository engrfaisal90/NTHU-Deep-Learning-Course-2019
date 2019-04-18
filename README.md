# NTHU-Deep-Learning-Course-2019
This repository contain all the assignments of Graduate level course (deep learning of spring semester 2019 at NTHU.

Each assignment contain "Assignment.pdf" file , "Student_ID.pdf" file and code folder.
  Assignment.pdf file contain questions of the assignment.
  Student_ID.pdf is the report file and contain answers to the question and detail about the code.
  Code folder contain all the code to solve those questions. the code folder might additionaly contain "data" folder and "output files" folder
    Data folder contain all the related used in the assignment.
    output file contain the files generate in the assignment. 


##Assignmnet 1 code
  Although detail is included in the Assignment.pdf" file and "Student_ID.pdf" in assignment folder. Here is outline of the code.
  hw1 StudentID.py contain the main function.
  Question1.py. gives detail implemtaion of question 1 of the assignment.
  Question2.py. gives detail implementaion of question 2 of the assignment.
  Question3.py. gives detail implementaion of question 3 of the assignment.
  Preprocessing.py preprocess the data. it contains following functions
    One-hot-encoding ~ convert the features to one hot encoding.
    split-data ~ split the data into train and test data and also separate the label.
    normalize ~ it normalizes each feature according to requirement.
  Regression.py includes all the functions required in regression.
    Lreg ~ Implement a linear regression model without the bias term to predict G3.   if lambda =0 is passed 
           Implement a regularized linear regression model without the bias.  if lambda value is passed.
    LReg_with_bias_reg ~ Implement a regularized linear regression model without the bias.
    Bayesian_LR ~ Implement a Bayesian linear regression model with the bias term.
    Logistic_regression ~ Implement a Logistic_regression model.
    Gradient_descent ~ Implements a function that calculates gradient desscent.
  Evaluate.py includes the function that evaluates the algorithms.
    get_confusion_matrix calculates True positive, True negative, False positive and false negative values
    Accuracy_precision calulayes Accuracy and precision.
  Plot.py plots all the required figures in the assignment.
    all_reg ~ plots ground truth values against all models.
    plot_confusion_matrix plot the confusion matrix.
    
