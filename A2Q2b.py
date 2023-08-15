#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np

import matplotlib.pyplot as plt
data_set = pd.read_csv("A2Q2Data_train.csv", header = None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
train_set = data_set.iloc[:,0:100]
label = data_set.iloc[:,100]
label = label.to_numpy()
train_set = train_set.to_numpy()


# In[2]:


train_set = np.transpose(train_set)
train_set_transpose = np.transpose(train_set)
Cov_mat = np.matmul(train_set,train_set_transpose)
pseudo_inverse = np.linalg.pinv(Cov_mat)
mid_term = np.matmul(pseudo_inverse,train_set)
max_likelihood = np.matmul(mid_term,label)


# In[3]:


train_set_transpose = np.transpose(train_set)
param_1 = np.matmul(train_set,train_set_transpose)
param_2 = np.matmul(train_set, label)


# In[8]:


w = np.zeros(100)
for i in range(1000):
    #print("before: ")
    grad = (2*np.matmul(param_1,w)-2*param_2)/np.linalg.norm(2*np.matmul(param_1,w)-2*param_2)
    #print(grad)
    w = w - ((2*np.matmul(param_1,w)-2*param_2))/i
    #print(np.linalg.norm(w-max_likelihood))


# In[17]:


def grad_descent(T,param_1,param_2):
    diff = []
    summer = 0
    w = (np.zeros(100))
    for i in range(1,T+1):
        w = w - ((2*np.matmul(param_1,w)-2*param_2)/np.linalg.norm(2*np.matmul(param_1,w)-2*param_2)**2)/i
        diff.append(np.linalg.norm(w-max_likelihood))
    #print(diff)
    #print(w)
    plt.figure(figsize=(10, 7))
    plt.title("plot showing convergence of w into wml")
    plt.xlabel("number of iterations")
    plt.ylabel("difference between gradient descent w and wml") 
    plt.plot(diff)


# In[25]:


#max_likelihood


# In[18]:


import random
grad_descent(1000, param_1, param_2)


# In[19]:


w

