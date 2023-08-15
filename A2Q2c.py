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


# In[6]:


def stoch_grad_descent(T, train_set, label):
    w = np.zeros(100)
    result = np.zeros(100)
    diff = []
    for i in range(1,T+1):
        mini_batch = np.random.choice(a=10000, size = 100, replace = False)
        train_set_tap = np.zeros((100,100))
        label_tap = np.zeros(100)
        
        k = 0
        index = 0
        for i in range(100):
            index = mini_batch[i]
            train_set_tap[:,i] = train_set[:,index]
            label_tap[i] = label[index]
        train_set_tap_transpose = np.transpose(train_set_tap)
        param_1 = np.matmul(train_set_tap,train_set_tap_transpose) 
        param_2 = np.matmul(train_set_tap, label_tap)
        w = w - ((2*np.matmul(param_1,w)-2*param_2)/np.linalg.norm(2*np.matmul(param_1,w)-2*param_2))/i
        result += w

        diff.append(np.linalg.norm(w-max_likelihood))
    plt.figure(figsize=(20, 20))
    print(diff[T-1])
    plt.figure(figsize=(10, 7))
    plt.xlabel("number of iterations")
    plt.ylabel("difference of w and wml")   
    plt.title("stichastic gradient descent converging to wml")
    plt.plot(diff)
    return result/100


# In[7]:


result = stoch_grad_descent(2000, train_set, label)
#print(result/2000)

