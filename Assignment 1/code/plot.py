"""
Created on Thu Mar 21 15:43:59 2019

@author: faisalg

"""
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def all_reg(test_Y,ypred_nobias_noreg,c_nobias_noreg,ypred_reg_nobias,c_reg_nobias,ypred_reg_bias,c_reg_bias,ypred_bayesian,c_bayesian):
    
    plt.plot(test_Y,color='b')

    plt.plot(ypred_nobias_noreg, color='y')
    plt.plot(ypred_reg_nobias, color='g')
    plt.plot(ypred_reg_bias, color = 'c')
    plt.plot(ypred_bayesian, color = 'm')
    lp='('+str(c_nobias_noreg.round(6))+') Linear Regression'
    rp='('+str(c_reg_nobias.round(6))+') Linear Regression (reg)'
    lb='('+str(c_reg_bias.round(6))+') Linear Regression (r/b)'
    ba='('+str(c_bayesian.round(6))+') bayesian Linear Regression'
    plt.legend(('Ground Truth',lp, rp, lb, ba),loc='lower right', shadow=True)
    
    plt.xlabel("Data Sample index")
    plt.ylabel("value of G3")
    plt.figure()
    plt.show()
    
    
def plot_confusion_matrix(values,acc,pre,thres):
    plt.figure()
    TP=values[0]
    FP=values[1]
    TN=values[2]
    FN=values[3]
    d = {'1': [FN, TP], '0': [TN, FP]}
    df = pd.DataFrame(data=d)
    sns.set(font_scale=.8)#for label size
    sns.heatmap(df, annot=True,annot_kws={"size": 14}, fmt='g',cmap="YlGnBu")# font size
    plt.xlabel('True')
    plt.ylabel('Predicted')
    tit='Accuracy= '+str(acc)+'     Precision= '+str(pre)+'     Threshold= '+str(thres)
    plt.title(tit,size=16)
   
    