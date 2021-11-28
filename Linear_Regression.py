import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'E:\chen yuanyao\Machine Learning\ex1-ex8-matlab\ex1\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['population','profit'])
data.head()
data.describe()

data.plot(kind='scatter', x='population', y='profit', figsize=(8,5))
#plt.show()
#insert one column to vectorization data
data.insert(0,'ones',1)
#initialization variables
columns = data.shape[1] #列数
X = data.iloc[:, 0 : columns - 1]#前columns-1列为输入向量
y = data.iloc[:, columns - 1 : columns]#最后一列为目标向量
X.head()
y.head()

#gradient descent algorithm to compute cost J(theta)=1/2m*sum(htheta(xi)-yi)^2
def cost(X,y,theta):
    m = len(X)
    inner_part = np.power(((X * theta.T) - y), 2)
    cost = np.sum(inner_part)/(2 * m)
    return cost
#Find theta
def GradientDescent(X,y,theta,alpha,epoch):
    #temp = np.matrix(np.zeros(theta.shape()))
    temp = np.ones((1, 2))
    parameters = int(theta.flatten().shape[1])
    cost1 = np.zeros(epoch)
    m = X.shape[0] #X(97,2) get the number of sample

    error = (X * theta.T) - y
    #for j in range(parameters):
        #term = np.multiply(error,X[:, j])
        #temp[0,j] = theta[0,j] - (alpha/m) * np.sum(term)
        #theta = temp
        #cost1[j] = cost(X,y,theta)
    for i in range(epoch):
        temp = theta - (alpha/m) * ((X * theta.T) - y).T * X
        theta = temp
        cost1[i] = cost(X,y,theta)

    return theta,cost1
#normal equation
def NormalEquation(X,y):
    theta = np.linalg.inv(X.T @ X)@ X.T@y
    return theta
#initialization theta and epoch
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])
alpha = 0.01
epoch = 1000
final_theta,final_cost = GradientDescent(X,y,theta,alpha,epoch)
normal_theta,normal_cost = NormalEquation(X,y)
cost(X,y,final_theta)
#plot the result
x = np.linspace(data.population.min(),data.population.max())# x
f = final_theta[0,0] + (final_theta[0,1] * x)# y
print(normal_theta)
n = (-normal_theta[0,0] * x)

fig, ax = plt.subplots(figsize = (6,4))
ax.plot(x,f,'r',label='linerat regression')
ax.scatter(data['population'],data.profit, label = 'traning data')
ax.legend(loc = 2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('linear regression')
#plt.show()
fig, ax = plt.subplots(figsize = (6,4))
ax.plot(x,n,'r',label='linear regression')
ax.scatter(data['population'],data.profit, label = 'traning data')
ax.legend(loc = 2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('linear regression')
plt.show()
#gradient graph

fig, ax = plt.subplots(figsize = (8,4))
ax.plot(np.arange(epoch), final_cost, 'r')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
ax.set_title('gradient descent')
plt.show()











