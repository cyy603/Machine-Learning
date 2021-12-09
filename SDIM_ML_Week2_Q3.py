import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path1 = 'E:\\chen yuanyao\\Machine Learning\\SDM\\Week2\\Homework2\\traindata.txt'
path2 = 'E:\\chen yuanyao\\Machine Learning\\SDM\\Week2\\Homework2\\testdata.txt'

data1 = pd.read_csv(path1, header = None, names =['Size', 'Bedrooms', 'Price'])
data2 = pd.read_csv(path2, header = None, names =['Size', 'Bedrooms', 'Price'])

data1.insert(0, 'Ones', 1)
data2.insert(0, 'Ones', 1)

index = [1, 2, 3, 4, 5, 6]
min_error = 100000
best_index_x1 = 0
best_index_x2 = 0

def cost_function(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    cost = np.sum(inner / 2 * len(X))
    return cost

def gradient_descent(X, y, alpha, epoch):
    theta = np.ones((1, X.shape[1]))
    temp_cost = np.zeros(epoch) 
    for i in range(epoch):
        partial_theta = np.sum((((X @ theta.T) - y).T @ X) / len(X))
        temp = theta - alpha * partial_theta
        theta = temp
        temp_cost[i] = cost_function(X, y, theta)
    return theta, temp_cost

def My_Linear_Regression(X, y, alpha, epoch):
    theta, cost = gradient_descent(X, y, alpha, epoch)
    #Price = theta[0,0] + theta[0,1] * X['Size'] + theta[0,2] * X['Bedrooms']
    print(cost[9990])
    return theta

def predict(X, theta):
    Price = X @ theta.T
    return Price

def cal_error(data_train, data_test):
    X_train = data1.iloc[:, 0: data1.shape[1] - 1]
    y_train = data1.iloc[:,data1.shape[1] - 1 : data1.shape[1]]
    X_test = data2.iloc[:, 0: data2.shape[1] - 1]
    y_test = data2.iloc[:,data2.shape[1] - 1 : data2.shape[1]]

    X_train = np.matrix(X_train.values)
    y_train = np.matrix(y_train.values)
    X_test = np.matrix(X_test.values)
    y_test = np.matrix(y_test.values)
    
    Predict_theta = My_Linear_Regression(X_train, y_train, alpha = 0.00001, epoch = 10000)
    print(Predict_theta)
    Predict_Price = predict(X_test, Predict_theta)
    bias = np.power((Predict_Price - y_test), 2)
    error = 1 / 2 * np.mean(bias)
    return error

for i in index:
    for j in index:
        data1['Size'] = data1['Size'] ** i
        data2['Size'] = data2['Size'] ** i
        data1['Bedrooms'] = data1['Bedrooms'] ** j
        data2['Bedrooms'] = data2['Bedrooms'] ** j
        error = cal_error(data1, data2)

        if min_error > error:
            min_errpr = error
            best_index_x1 = i
            best_index_x2 = j

print(i,j,min_error)