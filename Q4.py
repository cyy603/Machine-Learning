import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def train(X, y):
    sk_lr = LinearRegression()
    sk_lr.fit(X, y)
    bias = np.power((sk_lr.predict(X) - y), 2)
    cost = 1 / 2 * np.mean(bias)
    return sk_lr, cost

def add_polymer(list,n):
    temp_list = list
    for i in range(n - 1):
        pow_list = np.power(temp_list, i)
        list = np.column_stack((list, pow_list))
    return list
def feature_scalling(X):
    return (X - X.mean()) / X.std()

path = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week2\\Homework2\\Weather.csv"
data = pd.DataFrame(pd.read_csv(path))

train_row = int(data.shape[0] * 0.8)
train_data = data.iloc[0 : train_row, :]
test_data = data.iloc[train_row : data.shape[0], :]

index = [1, 2, 3]
error_list = []

min_error = 100000
opt_index = 0
opt_coe = 0
a = 1

for i in index:
    train_X = copy.deepcopy(train_data["P-DM"])
    train_y = copy.deepcopy(train_data["T-DM"])
    test_X = copy.deepcopy(test_data["P-DM"])
    test_y = copy.deepcopy(test_data["T-DM"])

    train_X = np.array([train_X.values]).T
    test_X = np.array([test_X]).T
    train_X = add_polymer(train_X, i)
    test_X = add_polymer(test_X, i)
    train_X = feature_scalling(train_X)
    train_y = feature_scalling(train_y)
    test_X = feature_scalling(test_X)
    test_y = feature_scalling(test_y)

    ones1 = np.ones(train_X.shape[0])
    train_X = np.column_stack((ones1, train_X))
    train_X = np.matrix(train_X)
    train_y = np.matrix(train_y)
    train_y = train_y.T

    ones2 = np.ones(test_X.shape[0])
    test_X = np.column_stack((ones2, test_X))
    test_X = np.matrix(test_X)
    test_y = np.matrix(test_y)
    test_y = test_y.T

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    sk_lr, cost = train(train_X, train_y)
    #train_y = train_y.T
    bias = np.multiply(sk_lr.predict(train_X) - train_y, sk_lr.predict(train_X) - train_y)
    error = 1 / 2 * np.mean(bias)
    error_list.append(error)
    if min_error > error:
        min_error = error
        min_bias = bias
        opt_index = i
        opt_coe = sk_lr.coef_
        min_cost = cost
        opt_sk_lr = sk_lr
        opt_X = test_X
        opt_y = test_y
test_bias = np.multiply(opt_sk_lr.predict(opt_X) - opt_y, opt_sk_lr.predict(opt_X) - opt_y)
test_error = 1 / 2 * np.mean(test_bias)

x = range(test_X.shape[0])
y = test_y.A1
plt.figure()
plt.scatter(x, y, label = "test data")
plt.plot(x, opt_sk_lr.predict(opt_X), label = 'predict')
plt.legend()

plt.show()
print(min_error, opt_index, opt_coe)
print(test_error)