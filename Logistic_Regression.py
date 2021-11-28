import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'E:\chen yuanyao\Machine Learning\ex1-ex8-matlab\ex2\ex2data1.txt'
data = pd.read_csv(path, header = None, names=['exam1','exam2','admitted'])
data.head()
data.describe()

positive = data[data.admitted.isin(['1'])]  
negetive = data[data.admitted.isin(['0'])]  

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)

ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#plt.show()

def sigmoid_function(z):
    return 1 / (1 + np.exp(- z))

x1 = np.arange(-10, 10, 0.1)
plt.plot(x1, sigmoid_function(x1), c='r')
#plt.show()

def cost_function(theta, X, y):
    a = y * np.log(sigmoid_function(X @ theta))
    b = (1 - y) * np.log(sigmoid_function(X @ theta))
    return np.mean(-(a + b))
#vactorization
#data.insert(0,1)
if 'Ones' not in data.columns:
    data.insert(0, 'Ones', 1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.zeros(X.shape[1])

#print(X.shape, theta.shape, y.shape)

cost = cost_function(theta,X,y)
#print(cost)

def Gradient_Descent(theta,X,y):
   return (X.T @ (sigmoid_function(X @ theta) - y))/len(X)  

#print(Gradient_Descent(theta,X,y))
#optimization

#prediction
def prediction(theta,X):
    probability = sigmoid_function(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]
result = Gradient_Descent(theta,X,y)

x1 = np.arange(130,0.1)
x2 = -(result[0] + result[1]*x1) / result[2]

#plot data
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='Admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x1, x2)
ax.set_xlim(0, 130)
ax.set_ylim(0, 130)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Decision Boundary')
#plt.show()
