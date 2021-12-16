import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

path = "E:\chen yuanyao\Machine Learning\SDM\Week2\Homework2\weather.csv"
data = pd.DataFrame(pd.read_csv(path))

train_row = int(data.shape[0] * 0.8)
train_data = data.iloc[0 : train_row, :]
test_data = data.iloc[train_row : data.shape[0], :]

index = [2]
error_list = []
min_error = 100000
opt_index = 0

def train(X, y):
    sk_lr = LinearRegression()
    sk_lr.fit(X, y)
    bias = np.power((sk_lr.predict(X) - y), 2)
    cost = 1 / 2 * np.mean(bias)
    return sk_lr, cost

def add_polymer(list, n):
    temp_list = list
    """
    if n == 0.5:
        pow_list = np.power(temp_list, 0.5)
        list = np.column_stack((list, pow_list))
        return list
    """
    for i in range(n - 1):
        pow_list = temp_list ** i
        list = np.column_stack((list, pow_list))
    return list

def feature_scalling(X):
    return (X - X.mean()) / X.std()

def pre_process(dataframe):
    temp1 = dataframe["P-DM"]
    temp1 = np.array([temp1.values]).T
    temp2 = dataframe["T-DM"]
    temp2 = np.array([temp2.values]).T
    temp = np.column_stack((temp1, temp2))
    temp3 = dataframe["RH-Mean"]
    temp3 = np.array([temp3.values]).T
    temp = np.column_stack((temp, temp3))
    return temp

for i in index:
    train_X = copy.deepcopy(train_data[["P-DM", "T-DM", "RH-Mean"]])
    train_y = copy.deepcopy(train_data["Sun-hours"])
    test_X = copy.deepcopy(test_data[["P-DM", "T-DM", "RH-Mean"]])
    test_y = copy.deepcopy(test_data["Sun-hours"])

    #train_X = np.array([train_X.values])
    #test_X = np.array([test_X])
    train_X = pre_process(train_X)
    test_X = pre_process(test_X)

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

print(min_error, opt_index, opt_coe)
print(test_error)

x = range(test_X.shape[0])
y = test_y.A1
plt.scatter(x, y, label = "test data")
plt.plot(x, opt_sk_lr.predict(test_X), label = "predict")
plt.legend()
plt.show()
