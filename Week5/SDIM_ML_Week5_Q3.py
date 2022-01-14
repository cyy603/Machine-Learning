import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def read_image(path):
    num = len(glob.glob(pathname = path + '/*.jpg'))
    X = np.zeros((num, 16 * 16))
    i = 0
    for filename in os.listdir(path):
        img_path = path + '/' + filename
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(src = img, dsize = (16, 16), interpolation = cv2.INTER_AREA)
        img = np.ravel(img)
        X[i, :] = img
        i += 1

    return X, num

def read_data(path):
    X, num = read_image(path + '/0')
    y = np.zeros((num, 1))
    for i in range(1, 10):
        temp_X, num = read_image(path + '/' + str(i))
        temp_y = np.zeros((num, 1)) + i
        X = np.row_stack((X, temp_X))
        y = np.row_stack((y, temp_y))

    return X, y

def cross_validation(data, start, end, split):
    test_set = data[start * split : end * split]
    test_index = list(test_set.index)
    test_flag = data.index.isin(test_index)
    train_flag = [not f for f in test_flag]
    train_set = data[train_flag]

    return train_set, test_set

def train(df_train, d_in, d_out, step_size, epoch):
    train_set = df_train.values
    train_X = torch.tensor(train_set[:, 0 : train_set.shape[1] - 1]).float()
    train_y = torch.tensor(train_set[:, train_set.shape[1] - 1 : train_set.shape[1]].ravel()).long()
    Net = nn.Sequential(
        nn.Linear(d_in, 800),
        nn.ReLU(),
        nn.Linear(800, 400),
        nn.ReLU(),
        nn.Linear(400, 100),
        nn.ReLU(),
        nn.Linear(100, d_out)
    )
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(Net.parameters(), lr = step_size)
    for i in range(epoch):
        y_hat = Net(train_X)
        loss = loss_func(y_hat, train_y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return Net

def evaluate(Net, test_set):
    test_set = test_set.values
    test_X = torch.tensor(test_set[:, 0 : test_set.shape[1] - 1]).float()
    test_y = test_set[: , test_set.shape[1] - 1 : test_set.shape[1]].ravel()
    test = Net(test_X)
    predict = torch.max(test, 1)[1].data.numpy().ravel()
    accuracy = accuracy_score(test_y, predict)

    return accuracy

if __name__ == '__main__':
    accuracies = []
    folds = 10
    path = ['E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\mnist\\train',
            'E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\mnist\\test',
            'E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\syn\\train',
            'E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\syn\\test']
    X, y = read_data(path[0])
    for i in range(1,4):
        temp_X, temp_y = read_data(path[i])
        X = np.row_stack((temp_X, X))
        y = np.row_stack((temp_y, y))
    val_data = np.column_stack((X, y))
    split = int(val_data.shape[0] / folds)
    df_data = pd.DataFrame(val_data)
    for i in range(1, 11):
        print(f'index = {i}')
        df_train, df_test = cross_validation(df_data, start = i - 1, end = i, split = split)
        Net = train(df_train = df_train, d_in = df_train.shape[1] - 1, d_out= 10, step_size = 0.001, epoch = 500)
        accuracy = evaluate(Net = Net, test_set = df_test)
        accuracies.append(accuracy)
        print(f'accuracy = {accuracy}')
    final_acc = np.mean(accuracies)
    print(final_acc)