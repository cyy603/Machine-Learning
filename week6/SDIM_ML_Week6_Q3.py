import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import os
import cv2
import glob

class SVM():
    def __init__(self, c, sigma, d, b, kernel):
        self.c = c
        self.sigma = sigma
        self.d = d
        self.b = b
        self.kernel = kernel


    def Kernel(self, X1, X2):
        #Setting gaussian kernel as default kernel function
        if self.kernel == 'gaus':
            return np.exp(- ((X1 - X2) ** 2).sum() / (2 * self.sigma ** 2))

        if self.kernel == 'linear':
            return np.dot(X1, X2)

        if self.kernel == 'poly':
            return np.power(self.sigma * (X1 @ X2) + self.b, self.d)

        if self.kernel == 'sigmoid':
            vira = self.sigma * (X1 @ X2) + self.b
            return np.tanh(vira)


    def train(self, X, y):
        self.m = X.shape[0]
        self.X = X
        self.y = y
        e = 1e-3

        #calculate alpha for SVM
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                P[i][j] = y[i] * y[j] * self.Kernel(np.array(X[i, :]), np.array(X[j, :]))

        P = matrix(P)
        Q = matrix(-np.ones((self.m, 1)))
        G = matrix(np.vstack((-np.identity(self.m), np.identity(self.m))))
        H = matrix(np.vstack((np.zeros((self.m, 1)), self.c * np.ones((self.m, 1)))))
        A = matrix(np.array(y, dtype = float).reshape(1, -1))
        b = matrix(0.0)

        sol = solvers.qp(P = P, q = Q, G = G, h = H, A = A, b = b)
        alpha = sol['x']
        self.alpha = np.array(alpha)
        self.b = 0

        #calculate b for SVM
        #e is the error that produced after the solver established the optimal problem
        for i , a in enumerate(self.alpha):
            if a - e > 0 and a < self.c - e:
                self.b = (1 - (self.alpha.T @ P[:, i])) / y[i]
                break
            else:
                continue

        return

    def predict(self, X):
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            res = self.b
            for j in range(0, self.m):
                res += self.alpha[j] * self.y[j] * self.Kernel(self.X[j, :], X[i, :])

            if res >= 0:
                y[i] = 1
            else:
                y[i] = -1

        return y

def read_image(path):
    num = len(glob.glob(pathname = path + '/*.jpg'))
    X = np.zeros((num, 28*28))
    i = 0
    for filename in os.listdir(path):
        img_path = path + '/' + filename
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        features = np.ravel(img)
        X[i, :] = features
        i += 1
    return X, num

def read_data(path):
    X, num0 = read_image(path + '/1')
    y = np.ones((num0, 1))
    for i in range(2, 3):
        temp_X, num = read_image(path + '/' + str(i))
        temp_y = np.zeros((num, 1)) + -1
        X = np.row_stack((X, temp_X))
        y = np.row_stack((y, temp_y))
    return X, y

if __name__ == '__main__':
    path_train = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week6\\Homework6\\data\\Digits\\train"
    path_test = "E:\\chen yuanyao\\Machine Learning\\SDM\\Week6\\Homework6\\data\\Digits\\test"
    train_X, train_y  = read_data(path_train)
    test_X, test_y = read_data(path_test)

    model = SVM(c = 15, sigma = 0.01, d = 1.0, b = 0.1, kernel = 'poly')
    model.train(train_X, train_y)
    pred = model.predict(test_X)
    print(classification_report(test_y, pred))
