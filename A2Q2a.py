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


# In[8]:


print(max_likelihood)


# In[3]:


testdat_set = pd.read_csv("A2Q2Data_test.csv", header = None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
testx = testdat_set.iloc[:,0:100]
testy = testdat_set.iloc[:,100]
testy = testy.to_numpy()
testx = testx.to_numpy()
def error_on_test(testx, testy ,w):
    error = 0
    error = np.matmul(testx,w) - testy
    return error
print(np.linalg.norm(error_on_test(testx,testy, max_likelihood))**2)

