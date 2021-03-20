#!/usr/bin/env python
# coding: utf-8

# In[1]:

##importing all the required libraries
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy

import sys
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

import matplotlib.pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:

## csv file read
train_df=pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\train.csv", encoding="UTF-8", header=None)
test_df=pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\test.csv", encoding="UTF-8",header=None)

train_df.head()


# In[3]:

## converting binary classification to -1 from 0. letting the others be equal to +1
def convert(element):
    
    if(element == 0.0):
        element-=1
        return element
    return element

train_df.iloc[:,0]=train_df.iloc[:,0].apply(convert) 
test_df.iloc[:,0]=test_df.iloc[:,0].apply(convert) 

train_df.head()


# In[4]:


columns = list(train_df.columns)


# In[8]:

##grid search cross validation

from sklearn.model_selection import cross_val_score
from sklearn import svm
##hyper parameters chosen from 1 to 100 as multiles of 5
hyper_params = [x*5 for x in range(1,20)]
accuracy = []
for hp in hyper_params:
    ## using kfold cross validation with k = 5
    clf=svm.SVC(C=hp )
    scores = cross_val_score(clf, X=train_df[columns[1:]],y=train_df.iloc[:,0], scoring = 'accuracy', cv = 5)
    accuracy.append(numpy.mean(scores))
## printing all the accuracies as a list of values
print(accuracy)


# In[5]:

## dividing labels and samples

y_train=train_df[0]
X_train=train_df.iloc[:,1:]

X_test=test_df.iloc[:,1:]
y_test=test_df[0]

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


# In[6]:


n_samples , n_features=X_train.shape
print(n_samples,n_features)


# In[7]:

## converting datasets into matrices using to_numpy()
y_train.shape
XTrain = X_train.to_numpy()
yTrain = y_train.to_numpy()


XTest = X_test.to_numpy()
yTest = y_test.to_numpy()

print(XTrain)


# In[8]:

import numpy as np

## model training method
def svm_train_primal(yTrain,XTrain, C):
    
    n_samples,n_features = XTrain.shape
    ##normalization of hyperparameter
    C=C/n_samples
    
    ## finding a matrix m which is the dot product of ith and jth features
    m=np.zeros((n_samples,n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            m[i,j] = np.dot(XTrain[i], XTrain[j])
    
    ## using cvxopt to make variable matrices
    
    P=cvxopt_matrix(np.outer(yTrain,yTrain)*m)
    q=cvxopt_matrix(np.ones(n_samples)*-1)
    A=cvxopt_matrix(yTrain,(1,n_samples))
    b=cvxopt_matrix(0.0)
    G=cvxopt_matrix(np.vstack((np.diag(np.ones(n_samples)*-1), np.identity(n_samples))))
    h = cvxopt_matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    
    solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    
    ## lagrange multiplier by solving the quadratic equation
    alphas = np.ravel(solver['x'])
    sv = alphas > 1e-5
    S = (alphas[sv]).flatten()
    w = ((yTrain * alphas).T @ XTrain).reshape(-1,1)
    b = np.mean(yTrain - np.dot(XTrain, w))
    
    return w, b

w_values_primal,b_primal = svm_train_primal(yTrain,XTrain,10)


# In[9]:

## method to calculate accuracy
def accuracy_model(yTest,XTest,w, b):
    
    y_values_predicted = np.sign(np.dot(XTest, w) + b)
    
    correct =0
    incorrect =0
    for i in range(0,1500):
        if(y_values_predicted[i]==yTest[i]):
            correct+=1
        else :
            incorrect+=1
    accuracy = (correct)/1500
    return accuracy

##accuracy for primal
test_accuracy = accuracy_model(yTest,XTest,w_values_primal,b_primal)
print(test_accuracy)


# In[10]:


## rechecking our predicted accuracy method using metrics from scikit learn library

y_values_predicted_primal = np.sign(np.dot(XTest, w_values_primal) + b_primal)
from sklearn import metrics
metrics.accuracy_score(yTest, y_values_predicted_primal)


# In[11]:


import numpy as np
def svm_train_dual(yTrain,XTrain,c):
    
    import numpy as np
    #Initializing values and computing H. Note the 1. to force to float type
    
    n_samples,n_features = XTrain.shape
    ##normalization of hyperparameter
    C=c/n_samples
    
    ## X_negate just multiplies -1 to all the features i.e negative of the features
    yTrain = yTrain.reshape(-1,1) * 1.
    X_negate = yTrain * XTrain
    M = np.dot(X_negate , X_negate.T) * 1.

    #using cvxopt to compute matrices
    P = cvxopt_matrix(M)
    q = cvxopt_matrix(-np.ones((n_samples, 1)))
    A = cvxopt_matrix(yTrain.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    G = cvxopt_matrix(np.vstack((np.eye(n_samples)*-1,np.eye(n_samples))))
    h = cvxopt_matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))

    #Run solver solving the quadratic equation
    solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solver['x'])

    ## finding w and b
    w = ((yTrain * alphas).T @ XTrain).reshape(-1,1)
    w = w.flatten()
    S = (alphas > 1e-5).flatten()
    b = np.mean(yTrain[S] - np.dot(XTrain[S], w))

    return w, b
## w and b values for dual
w_values_dual, b_dual = svm_train_dual(yTrain,XTrain,c=10)
    


# In[12]:

##calculating accuracy for dual
test_accuracy_d = accuracy_model(yTest,XTest,w_values_dual,b_dual)
print(test_accuracy_d)


# In[13]:

##checking accuracy using 3rd party implementation like scikit
from sklearn import metrics
from sklearn.svm import SVC

clf = SVC(C = 10/n_samples, kernel = 'linear')

##fitting the model
clf.fit(XTrain, yTrain.ravel()) 
pred = clf.predict(X_test)
w = clf.coef_
b = clf.intercept_
print(f'accuracy: {metrics.accuracy_score(y_test, pred)}')
##printing accuracy using metrics.accuracy_score from sklearn library
