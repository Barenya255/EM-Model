import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
import random
data_set = pd.read_csv("A2Q1.csv", header = None)
data_set = data_set.to_numpy()
data_set = np.transpose(data_set)

def initialize(K):
    assign = [np.random.randint(0,K) for i in range(len(data_set[0]))]
    return assign
def assign(X,K):
    #print(len(X[0]))
    assignment= np.zeros(len(X[0]), dtype = int)
    for i in range(len(X[0])):
        assignment[i] = random.randint(0,K-1)
    return assignment
def means(X,assignment,K):
    means = np.zeros((len(X),K))
    for i in range(len(X[0])):
        means[:,assignment[i]]+=X[:,i]
    numbers = np.zeros(K)
    for i in range(len(X[0])):
        numbers[assignment[i]]+=1
    means /= numbers
    return means
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
    return assignment

def init_lam(data_set, assignment, K):
    lam = np.zeros((400,4))
    for i in range(400):
        for j in range(50):
            lam[i][assignment[i]] =1
    return lam

def log_likelihood(data_set, mean, omega, K):
    log_likelihood = 0
    for i in range(len(data_set[0])):
        summ = 0
        for k in range(K):
            summ += omega[k]*pmf(data_set[:,i], mean[:,k])
        log_likelihood += np.log(summ)
    return log_likelihood
def expectation(data_set, mean, omega, k):
    lam = np.zeros((4,400), dtype = 'float')
    for i in range(k):
        for j in range(len(data_set[0])):
            lam[i][j] = pmf(data_set[:,j], mean[:,i]) * omega[i]
    summ = lam.sum(axis = 0)
    lam = lam/summ
    return np.transpose(lam)
def maximization(lam, K ,data_set):
    omega = np.zeros(K, dtype = 'float')
    N = 400
    lam = np.transpose(lam)
    #print(lam)
    mean = np.zeros((50,K), dtype = 'float')
    for k in range(K):
        summ = np.zeros(50, dtype = 'float')
        Nk = 0
        for i in range(400):
            Nk+=lam[k][i]
        omega[k] = Nk/N
        for i in range(400):
            summ = summ + lam[k][i]*data_set[:,i]
        mean[:,k] = summ/Nk
    return mean, omega
def pmf(data, mean):
    bern = 1
    #print(mean.shape)
    #print(data.shape)
    for d in range(50):
        #print(mean.shape)
        bern =  bern*(mean[d]**data[d])*((1-mean[d])**(1-data[d]))
    return bern
def EM(graph, K):
    assign = K_means(data_set,4)
    #print(assign)
    prev = 0.0
    lam = init_lam(data_set, assign, 4)
    mean, omega = maximization(lam, 4, data_set)
    curr = log_likelihood(data_set, mean, omega, K)
    while abs(curr-prev) >= 1e-2:
        prev = curr
        lam = expectation(data_set, mean, omega, 4)
        #print(omega)
        mean, omega = maximization(lam, 4, data_set)
        curr = log_likelihood(data_set, mean, omega, K)
        graph.append(curr)
        #print(curr)
    final = np.argmax(lam, axis = 1)
    return final,graph
import random
graph = []
for i in range(100):
    print(i, end = ", ")
    graph1 = []
    final, graph1 = EM(graph1,4)
    graph.append(graph1)
    
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

plt.figure(figsize=(10, 7))
plt.xlabel("number of iterations")
plt.ylabel("log likelihood values")   
plt.plot(result)


plt.figure(figsize=(10, 7))
plt.xlabel("cluster number")
plt.title("number of points per cluster")
plt.ylabel("number of points")   

summ = np.zeros(4)

for i in range(400):
    summ[final[i]]+=1
for i in range(400):
    plt.scatter(i,final[i])
error = 0.0
mu = np.zeros((50,4))
number = np.zeros(4)
for i in range(len(data_set[0])):
    number[final[i]]+=1
    mu[:,final[i]] += data_set[:,i]
for i in range(len(data_set[0])):
    error += np.linalg.norm(data_set[:,i]-mu[:,final[i]])**2
print(error)