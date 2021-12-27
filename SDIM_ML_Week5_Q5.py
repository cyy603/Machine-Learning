import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from PIL import Image

def read_the_image(path):
    m = Image.open(path)
    return m

def iterate(path, form, m):
    if m >= 0 and m <= 9:
        temp = "\\00000"
    if m >= 10 and m <= 99:
        temp = "\\0000"
    if m >= 100 and m <= 999:
        temp = "\\000"
    number = str(m)
    index = temp + number + form
    path = path + index
    return path

def vectorization(m):
    g, b, k = m.split()
    g = np.array(g).reshape((1, 784))
    b = np.array(b).reshape((1, 784))
    k = np.array(k).reshape((1, 784))
    image = np.hstack((g, b, k))
    return image


def preprocess_n(path, form, number):
    temp_path = iterate(path, form, m = 0)
    temp_image = read_the_image(temp_path)
    data = vectorization(temp_image)

    for i in range(number - 1):
        temp_path = iterate(path, form, m = i + 1)
        temp_image = read_the_image(temp_path)
        list_image = vectorization(temp_image)
        data = np.vstack((data, list_image))
    return data

def devide_xy(data):
    X = data[:, 0 : data.shape[1] - 1]
    y = data[:, data.shape[1] - 1 : data.shape[1]]
    return X, y

path1 = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week3\\Homework3\\data\\Digits\\train\\1"
path2 = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week3\\Homework3\\data\\Digits\\train\\2"
test_path1 = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week3\\Homework3\\data\\Digits\\test\\1"
test_path2 = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week3\\Homework3\\data\\Digits\\test\\2"
index = "\\00000"
form = ".jpg"

data1 = preprocess_n(path1, form, number = 479)
data2 = preprocess_n(path2, form, number = 479)
test_data1 = preprocess_n(test_path1, form, number = 119)
test_data2 = preprocess_n(test_path2, form, number = 119)
train_X = np.vstack((data1, data2))
test_X = np.vstack((test_data1, test_data2))
zeros1 = np.zeros((479, 1))
ones1 = np.ones((479, 1))
train_y = np.vstack((zeros1, ones1))
zeros2 = np.zeros((119, 1))
ones2 = np.ones((119, 1))
test_y = np.vstack((zeros2, ones2))

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
