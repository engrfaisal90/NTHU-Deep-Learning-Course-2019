# NTHU-Deep-Learning-Course-2019
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
  Although detail is included in the Assignment.pdf" file and "Student_ID.pdf" in assignment folder. Here is outline of the code.</br>
  * hw1 StudentID.py contain the main function.</br>
  * 
