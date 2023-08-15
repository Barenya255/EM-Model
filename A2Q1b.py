#!/usr/bin/env python
# coding: utf-8

# In[4]:

#importing libraries and the data set
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_set = pd.read_csv("A2Q1.csv", header = None)
data_set = data_set.to_numpy()
data_set = np.transpose(data_set)


# In[5]:

# The following functions reassign_vals, K_means, assign and means are used to perform Kmeans initialization which will then later be
#used to start up the EM algorithm
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


# In[6]:


def K_means(inp,K, mean):
    error_list = []
    check = True
    maxim = 0
    assignment = assign(inp,K)
    while maxim< 500 and check == True:
        maxim += 1
        mean = means(inp,assignment,K,mean)
        check, error,assignment = reassign_vals(inp,mean,assignment,K)
        error_list.append(error)
    return assignment


# In[7]:


def assign(X,K):
    #print(len(X[0]))
    assignment= np.zeros(len(X[0]), dtype = int)
    for i in range(len(X[0])):
        assignment[i] = random.randint(0,K-1)
    return assignment


# In[8]:


def means(X,assignment,K, mean):
    means = np.zeros((len(X),K))
    for i in range(len(X[0])):
        means[:,assignment[i]]+=X[:,i]
    numbers = np.zeros(K)
    for i in range(len(X[0])):
        numbers[assignment[i]]+=1
    means /= numbers
    return means


# In[9]:

# The following function updates omega values
def omegas(data_set, assignment, K, omega):
    omega = np.zeros(K)
    for i in range(len(data_set[0])):
        omega[assignment[i]] +=1
    omega /= len(data_set[0])
    return omega


# In[10]:


def pdet(v_cov):
    eig_val = np.linalg.eigvals(v_cov)
    return np.product(eig_val[eig_val>1e-9])


# In[11]:

#
import mpmath as mp
def dist(i,k, omega, mean, sig):
    gauss_numer = data_set[:,i] - mean[:,k]
    for i in range(len(sig[k])):
        sig[k][i][i] += float(1e-3)
    if np.linalg.det(sig[k]) == 0:
        for i in range(len(sig[k])):
            sig[k][i][i] += random.random()
    siginv = np.linalg.inv(sig[k])
    gauss_numer1 = np.matmul(np.transpose(gauss_numer), siginv)
    gauss_numer = np.matmul(np.transpose(gauss_numer1),gauss_numer)
    gauss_numer = -0.5*gauss_numer
    gauss_numer =  float(mp.exp(gauss_numer))
    gauss = pow(6.28318530718,-25)*gauss_numer
    gauss = gauss/np.linalg.det(sig[k])**0.5
    return gauss


# In[28]:

#function that makes the variance covariance matrix
def sig_matrix(data_set, assignment, means):
    data_set_divide = [[] for i in range(4)]
    for i in range(len(data_set[0])):
        data_set_divide[assignment[i]].append(data_set[:,i])
    for i in range(4):
        data_set_divide[i] = np.array(data_set_divide[i], dtype = 'float')
        data_set_divide[i] = data_set_divide[i]
    for i in range(4):
        data_set_divide[i] = np.matmul(np.transpose(data_set_divide[i]),data_set_divide[i])
    return data_set_divide


# In[ ]:


# import math
# import random
# graph = []
# lam,omega,mean,v_cov,graph = EM()
# z = []
# for i in range(400):
#     z.append(np.argmax(lam[i]))
# z = np.array(z)
# print(z)
# plt.plot(graph)


# In[14]:

#fucntion to make the responsibility functions

def lambda_calc(omega,mean,v_cov,lam):
    lam = np.zeros((400,4))
    temp = np.zeros(50)
    summ = 0
    for i in range(len(data_set[0])):
        summ = 0
        for k in range(4):
            temp = dist(i,k,omega, mean, v_cov)*omega[k]
            summ += temp
        for k in range(4):
            lam[i][k] = dist(i,k,omega,mean,v_cov)*omega[k]/summ
    return lam


# In[15]:

#function to initialize for EM()
def initialize(K):
    mean = np.zeros((50,4))
    omega = np.zeros(4)
    assignment =assign(data_set,K)
    mean = means(data_set, assignment,4, mean)
    omega = omegas(data_set, assignment, 4, omega)
    v_cov = sig_matrix(data_set,assignment, means)
   
    lam = np.zeros((400,K))
    lam = lambda_calc(omega,mean,v_cov,lam)
        
    return assignment,omega,mean,v_cov,lam


# In[16]:

#function to update means
def update_means(data_set, lam, K, mean):
    numer= 0.0
    denom = 0.0
    for i in range(K):
        for j in range(len(data_set[0])):
            denom += lam[j][i]
            numer += lam[j][i] * data_set[:,j]
        mean[:,i] = numer/denom
    return mean


# In[17]:

#function to update variance_covariance matrix
def update_sig_matrix(data_set, lam, K, mean):
    term = np.zeros((50,1))
    result = np.zeros((50,50))
    sigs = []
    for k in range(4):
        for i in range(len(data_set[0])):
            term = data_set[:,i] - mean[:,k]
            term = np.outer(term,term)
            term *= lam[i][k]
            result += term
        result /= np.sum(lam[:,k])
        sigs.append(result)
        result = np.zeros((50,50))
    return sigs


# In[18]:

#function to update omegas
def update_omegas(data_set, lam, K, omega):
    for i in range(K):
        summ = 0
        for j in range(len(data_set[0])):
            summ += lam[j][i]
        omega[i] = summ/len(data_set[0])
    return omega


# In[19]:

#
def expectation(omega, mean, v_cov, lam, K):
    lam = lambda_calc(omega,mean,v_cov,lam)
    return lam


# In[20]:


def maximization(omega, mean, v_cov, lam, K):
    mean = update_means(data_set, lam,4, mean)
    omega = update_omegas(data_set,lam, 4, omega)
    v_cov = update_sig_matrix(data_set,lam, K, mean)
    return omega, mean, v_cov


# In[21]:


def EM():
    graph1 = []
    assignment,omega,mean,v_cov,lam = initialize(4)
    prev_log = 0
    curr_log = log_likelihood(4, omega, mean, v_cov)
    graph1.append(curr_log)
    #print(abs(curr_log-prev_log))
    while(abs(curr_log-prev_log) >= 1):
        #print(abs(curr_log-prev_log))
        prev_log = curr_log
        omega, mean, v_cov = maximization(omega, mean, v_cov, lam, 4)
        lam = expectation(omega, mean, v_cov, lam, 4)
        curr_log = log_likelihood(4, omega, mean, v_cov)
        #print(abs(curr_log-prev_log))
        graph1.append(curr_log)
    return lam,omega,mean,v_cov, graph1


# In[22]:


def log_likelihood(K, omega, mean, v_cov):
    summ = 0
    for i in range(len(data_set[0])):
        temp = 0
        for k in range(K):
            temp += omega[k]*dist(i,k, omega, mean, v_cov)
        if temp < 0:
            print(temp)
        summ+=math.log(temp)
    return summ


# In[29]:


import random
graph = []
for i in range(100):
    print(i, end = ", ")
    graph1 = []
    lam,omega,mean,v_cov, graph1 = EM()
    graph.append(graph1)
np.sum(omega)


# In[30]:


maxim = 0
for i in range(len(graph)):
    maxim = max(maxim,len(graph[i]))
for i in range(len(graph)):
    for j in range(maxim-len(graph[i])):
        graph[i].append(0)
# calculating averages
result = []
summ=0
for i in range(len(graph[0])):
    for j in range(len(graph)):
        summ+=graph[j][i]
    summ/=100
    result.append(summ)

plt.plot(result)


# In[31]:


plt.figure(figsize=(10, 7))
plt.xlabel("number of iterations")
plt.ylabel("log likelihood values")   
plt.plot(result)


# In[32]:


x= []
plt.figure(figsize=(10, 7))
plt.xlabel("cluster number")
plt.title("number of points per cluster")
plt.ylabel("number of points")   
for i in range(len(lam)):
    x.append(np.argmax(lam[i]))
x = np.array(x)
for i in range(400):
    plt.bar(x[i],i)


# In[33]:


x


# In[34]:


error = 0.0
mu = np.zeros((50,4))
number = np.zeros(4)
for i in range(len(data_set[0])):
    number[x[i]]+=1
    mu[:,x[i]] += data_set[:,i]
for i in range(len(data_set[0])):
    error += np.linalg.norm(data_set[:,i]-mu[:,x[i]])**2
print(error)


# 
