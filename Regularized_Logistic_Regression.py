import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Logistic_Regression as LR
import scipy.optimize as opt

path = 'E:\chen yuanyao\Machine Learning\ex1-ex8-matlab\ex2\ex2data2.txt'
data2 = pd.read_csv(path, names=['Test 1', 'Test 2', 'Accepted'])
data2.head()

def plot_data():
    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()

#plot_data()

def feature_mapping(x1,x2,power):
    data = {}#initialize the dict
    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            data["f{}{}".format(i - j, j)] = np.power(x1, i-j) * np.power(x2, j)
    return pd.DataFrame(data)

x1 = data2['Test 1'].values
x2 = data2['Test 2'].values

_data2 = feature_mapping(x1, x2, power=6)
_data2.head()
#print(data2)
#print(_data2)

X = _data2.values
y = data2["Accepted"].values
theta = np.zeros(X.shape[1])

#punish
def Reg_Cost(theta,X,y,l=1):
    _theta = theta[1:]
    reg = (1 / (2 *len(X))) +(_theta @ _theta)
    return LR.cost_function(theta,X,y) + reg

#print(Reg_Cost(theta, X, y, l=1))

def Reg_Gradient(theta,X,y, l=1):
    reg = (1/len(X)) * theta
    reg[0] = 0
    return LR.Gradient_Descent(theta,X,y) + reg

#print(Reg_Gradient(theta,X,y, l=1))
"""
final_theta = Reg_Gradient(theta,X,y, l=1)

x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta
z = z.reshape(xx.shape)

plot_data()
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)
"""
result2 = opt.fmin_tnc(func=Reg_Cost, x0=theta, fprime=Reg_Gradient, args=(X, y, 2))
final_theta = result2[0]
predictions = LR.prediction(final_theta, X)

x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)

z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
z = z @ final_theta
z = z.reshape(xx.shape)

plot_data()
plt.contour(xx, yy, z, 0)
plt.ylim(-.8, 1.2)



