import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

path = "E:\chen yuanyao\Machine Learning\SDM\Week3\Homework3\data\Disease\data_2.xlsx"
data = pd.DataFrame(pd.read_excel(path))
data_X = data.drop(columns = ["A_S", "CEREBRAL_APOPLEXTY"], inplace = False)
y = np.array(data["CEREBRAL_APOPLEXTY"])

def corss_validation(data):
    train_row = int(data.shape[0] * 0.8)
    train_data = data.iloc[0: train_row, :]
    test_data = data.iloc[train_row: data.shape[0], :]
    return train_data, test_data

train_X, test_X = corss_validation(data_X)
length = np.floor(len(y) * 0.8)
length = int(length)
train_y = y[0 : length]
test_y = y[length : len(y)]

for i in range(1, 6):
    poly = PolynomialFeatures(degree= i, interaction_only = True)
    polyx = poly.fit_transform(train_X)
    polyx_t = poly.fit_transform(test_X)

    std = StandardScaler()
    polyx_std = std.fit_transform(polyx)
    polyx_t_std = std.fit_transform(polyx_t)
    lg = LogisticRegression(penalty = "l2", solver = "newton-cg", fit_intercept = False, max_iter = 5000)
    lg.fit(polyx_std, train_y)
    print(f'order = {i}, score of training set = {lg.score(polyx_std, train_y)}, score of testing set = {lg.score(polyx_t_std, test_y)}')

