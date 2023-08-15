#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data_set = pd.read_csv("A2Q1.csv", header = None)
data_set = data_set.to_numpy()
data_set = np.transpose(data_set)


# In[3]:


def assign(X,K):
    print(len(X[0]))
    assignment= np.zeros(len(X[0]), dtype = int)
    for i in range(len(X[0])):
        assignment[i] = random.randint(0,K-1)
    return assignment


# In[4]:


def means(X,assignment,K):
    means = np.zeros((len(X),K))
    for i in range(len(X[0])):
        means[:,assignment[i]]+=X[:,i]
    numbers = np.zeros(K)
    for i in range(len(X[0])):
        numbers[assignment[i]]+=1
    means /= numbers
    return means


# In[9]:


def K_means(inp,K):
    error_list = []
    check = True
    maxim = 0
    assignment = assign(inp,K)
    while maxim< 500 and check == True:
        maxim += 1
        mean = means(inp,assignment,K)
        check, error,assignment = reassign_vals(inp,mean,assignment,K)
        error_list.append(error)
    return error_list, assignment


# In[6]:


def reassign_vals(inp,mean,assi,K):
    error = 0.0
    flag = 0

    
    for i in range(len(inp[0])):
        comp = np.zeros(K)
        for j in range(K):
            comp[j] = np.linalg.norm(inp[:,i]-mean[:,j])**2
        temp = np.argmin(comp)
        if temp != assi[i]:
            error += temp
            flag = 1
            assi[i] = temp
    for i in range(len(inp[0])):
        error += np.linalg.norm(inp[:,i]-mean[:,assi[i]])**2
    if flag == 1:
    #print("True")
        return True,error,assi
    else:
     #("False")
        return False,error,assi


# In[12]:


assign


# In[10]:


error_list, assign = K_means(data_set, 4)
plt.figure(figsize=(10, 7))
plt.xlabel("number of iterations")
plt.ylabel("log likelihood values")   
for i in range(400):
    plt.scatter(i,assign[i])


# In[11]:


import random
plt.figure(figsize=(10, 7))
plt.xlabel("number of iterations")
plt.ylabel("objective function value")   
plt.title("Kmeans convergence")
plt.plot(error_list, color = "red")


# In[15]:


error = 0.0
mu = np.zeros((50,4))
number = np.zeros(4)
for i in range(len(data_set[0])):
    number[assign[i]]+=1
    mu[:,assign[i]] += data_set[:,i]
for i in range(len(data_set[0])):
    error += np.linalg.norm(data_set[:,i]-mu[:,assign[i]])**2
print(error)

