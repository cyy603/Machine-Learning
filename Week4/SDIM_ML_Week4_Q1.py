import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
"""
def cross_validation(data):
    train_row = int(data.shape[0] * 0.8)
    train_data = data.iloc[0 : train_row, :]
    test_data = data.iloc[train_row : data.shape[0], :]
    return train_data, test_data
"""
def get_Data():
    train_row = int(data.shape[0] * 0.8)
    train_data = data.iloc[0: train_row, :]
    test_data = data.iloc[train_row: data.shape[0], :]
    train_X = train_data.iloc[:, 0: train_data.shape[1] - 1]
    train_y = train_data.iloc[:, train_data.shape[1] - 1: train_data.shape[1]]
    train_y = np.array(train_y.values).ravel()
    test_X = test_data.iloc[:, 0: test_data.shape[1] - 1]
    test_y = test_data.iloc[:, test_data.shape[1] - 1: test_data.shape[1]]
    test_y = np.array(test_y.values).ravel()
    return train_X, train_y, test_X, test_y

if __name__ == '__main__':
    path = "E:\chen yuanyao\Machine Learning\SDM\Week4\Homework4\Disease\data_1.xlsx"
    data = pd.DataFrame(pd.read_excel(path))
    #data.drop(columns=['BMI'], inplace=True)

    train_X, train_y, test_X, test_y = get_Data()
    for i in range(1, 10):
        poly = PolynomialFeatures(degree = i)
        polyx = poly.fit_transform(train_X)
        polyx_t = poly.fit_transform(test_X)

        std = StandardScaler()
        polyx_std = std.fit_transform(polyx)
        polyx_t_std = std.fit_transform(polyx_t)

        clf = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 5000, solver = 'lbfgs', alpha = 1)
        clf.fit(polyx_std, train_y)
        predict_y = clf.predict(polyx_std)
        predict_y_res = clf.predict(polyx_t_std)
        print(f'order - {i}\n'
              f'training result:\n'
              f'{classification_report(train_y, predict_y)}\n'
              f'test result:\n'
              f'{classification_report(test_y,predict_y_res)}')


