import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

def read_image(path):
    num = len(glob.glob(pathname = path +'/*.jpg'))
    X = np.zeros((num, 32*32))
    i = 0
    for filename in os.listdir(path):
        img_path = path + '/' + filename
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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

if __name__ == '__main__':
    d_out = 10
    step_size = 0.001
    momentum = 0.001
    epoch = 600
    train_X, train_y = read_data('E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\syn\\train')
    test_X, test_y = read_data('E:\\chen yuanyao\\Machine Learning\\SDM\\Week5\\Homework5\\data\\Digits\\syn\\test')
    d_in = train_X.shape[1]
    m = train_X.shape[0]
    train_X = torch.tensor(train_X).float()
    train_y = torch.tensor(train_y.ravel()).long()
    test_X = torch.tensor(test_X).float()
    test_y = np.ravel(test_y)
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
    optim = torch.optim.SGD(Net.parameters(), lr=step_size )
    for i in range(epoch):
        y_hat = Net(train_X)
        loss = loss_func(y_hat, train_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)

    test = Net(test_X)
    predict = torch.max(test, 1)[1].data.numpy().ravel()
    numbers = ['0', '1', '2', '3', '4 ', '5', '6', '7', '8', '9']
    print(classification_report(test_y, predict, target_names=numbers))