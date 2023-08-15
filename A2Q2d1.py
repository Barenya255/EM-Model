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


# In[3]:


train_set_transpose = np.transpose(train_set)
Cov_mat = np.matmul(train_set,train_set_transpose)
pseudo_inverse = np.linalg.pinv(Cov_mat)
mid_term = np.matmul(pseudo_inverse,train_set)
max_likelihood = np.matmul(mid_term,label)


# In[5]:





# In[4]:


training_set = train_set[:,0:8000]
label_set = label[:8000]
param_1 = np.matmul(training_set,np.transpose(training_set))
param_2 = np.matmul(training_set, label_set)


# In[5]:


def grad_descent(T,param_1,param_2, reg):
    #diff = []
    summer = 0
    w = (np.zeros(100))
    for i in range(1,T+1):
        grad = 2*np.matmul(param_1,w)-2*param_2+reg*w
        w = w - (1/i)*(grad)/np.linalg.norm(grad)
        #diff.append(np.linalg.norm(w-max_likelihood))
    return w
    #plt.plot(diff)


# In[6]:


def error(w,train_set,label):
    term = (np.matmul(np.transpose(train_set),w)-label)
    return np.linalg.norm(term)**2


# In[7]:


train_set[:,8000:10000].shape


# In[10]:


mini = 1000000
maxi = 0
lam = 0
prev_term = 10000
error_term = 10000
graph = []
val_set = train_set[:,8000:10000]
label_set_val = label[8000:]
for reg in range(20000): 
    w = grad_descent(1000, param_1, param_2, reg/1000)
    #prev_term = error_term
    error_term = error(w,val_set,label_set_val)
    graph.append(error_term)
    if error_term<mini:
        mini = error_term
        #print(mini)
        lam = reg/1000
        #print(lam)
        


# In[11]:



plt.figure(figsize=(10, 7))
plt.xlabel("number of iterations")
plt.ylabel("log likelihood values") 
plt.plot(graph)


# In[12]:


testdat_set = pd.read_csv("A2Q2Data_test.csv", header = None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
testx = testdat_set.iloc[:,0:100]
testy = testdat_set.iloc[:,100]
testy = testy.to_numpy()
testx = testx.to_numpy()


# In[13]:


def error_on_test(testx, testy ,w):
    error = 0
    error = np.matmul(testx,w) - testy
    return error


# In[14]:


print(np.linalg.norm(error_on_test(testx,testy, w))**2)
print(np.linalg.norm(error_on_test(testx,testy, max_likelihood))**2)


# In[15]:


print(w)

