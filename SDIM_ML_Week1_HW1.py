import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("E:\chen yuanyao\Machine Learning\SDM\Week1\homework1\所需数据与图片\data_1.csv")
row_x = np.array(data['x'])
row_y = np.array(data['y'])

#Vactorivation
#X = np.insert(row_x, 0, 1)
#y = np.insert(row_y, 0, 1)

X = row_x.reshape(101 ,1)
y = row_y.reshape(101 ,1)
def plot_data(x, y):
    plt.plot(x, y)
    plt.show()
#plot_data(x , y)
def cost_function(X, y, theta):
    inner = np.power(((X @ theta) - y), 2)
    cost = np.sum(inner / 2 * len(X))
    return cost

def gradient_descent(X, y, alpha, epoch):
    theta = np.ones((1, 2))
    temp_cost = np.zeros(epoch)
    for i in range(epoch):
        partial_theta = np.sum((((X @ theta) - y).T @ X) / len(X))
        temp = theta - alpha * partial_theta
        theta = temp
        temp_cost[i] = cost_function(X, y, theta)
    return theta, temp_cost

print(X.shape,y.shape)
epoch = 10
theta, cost = gradient_descent(X, y, alpha = 0.02, epoch = epoch)

x_value = np.linspace(data.x.min(), data.x.max())
y_value = theta[0,0] + theta[0,1] * x_value
x1 = np.linspace(data.x.min(), data.x.max())
y1 = cost
print(theta)
plt.plot(X,y)
plt.plot(x_value, y_value)

fig, ax = plt.subplots(figsize = (8,4))
ax.plot(np.arange(epoch), cost, 'r')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
ax.set_title('gradient descent')
plt.show()