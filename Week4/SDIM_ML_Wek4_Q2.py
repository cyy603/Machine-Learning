import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

path = "E:\chen yuanyao\Machine Learning\SDM\Week4\Homework4\Disease\data_2.xlsx"
data = pd.DataFrame(pd.read_excel(path))
y = np.array(data["CEREBRAL_APOPLEXTY"])
data.drop(columns = ["A_S", "CEREBRAL_APOPLEXTY"], inplace = True)


def cross_validation(data):
    train_row = int(data.shape[0] * 0.8)
    train_X = data.iloc[0: train_row, :]
    test_X = data.iloc[train_row: data.shape[0], :]
    return train_X, test_X

train_X, test_X = cross_validation(data)
length = np.floor(len(y) * 0.8)
length = int(length)
train_y = y[0 : length]
test_y = y[length : len(y)]

for i in range(1, 10):
    poly = PolynomialFeatures(degree=i)
    polyx = poly.fit_transform(train_X)
    polyx_t = poly.fit_transform(test_X)

    std = StandardScaler()
    polyx_std = std.fit_transform(polyx)
    polyx_t_std = std.fit_transform(polyx_t)

    mlp = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 5000, solver = 'lbfgs', alpha = 1)
    mlp.fit(polyx_std, train_y)
    predict_y = mlp.predict(polyx_std)
    predict_y_res = mlp.predict(polyx_t_std)
    print(f'order - {i}\n'
          f'training result:\n'
          f'{classification_report(train_y, predict_y)}\n'
          f'test result:\n'
          f'{classification_report(test_y, predict_y_res)}')