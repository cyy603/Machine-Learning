import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class SVM():
    def __init__(self, c, sigma, d, b, type):
        self.c = c
        self.sigma = sigma
        self.d = d
        self.b = b
        self.type = type


    def Kernel(self, X1, X2):
        #Setting gaussian kernel as default kernel function
        if self.type == 'gaus':
            return np.exp(- ((X1 - X2) ** 2).sum() / (2 * self.sigma ** 2))

        if self.type == 'linear':
            return np.dot(X1, X2)

        if self.type == 'poly':
            return np.power(self.sigma * (X1 @ X2) + self.b, self.d)

        if self.type == 'sigmoid':
            vira = self.sigma * (X1 @ X2) + self.b
            return (np.exp(vira) - np.exp(-vira)) / (np.exp(vira) + np.exp(-vira))


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

if __name__ == '__main__':
    path= 'E:\chen yuanyao\Machine Learning\SDM\Week6\Homework6\data\Disease\data_2.xlsx'
    train_data = pd.DataFrame(pd.read_excel(path, sheet_name = 'train'))
    test_data = pd.DataFrame(pd.read_excel(path, sheet_name = 'test'))
    train_data.loc[train_data.CEREBRAL_APOPLEXTY == 0] = -1
    test_data.loc[test_data.CEREBRAL_APOPLEXTY == 0] = -1
    train_X = train_data.iloc[:, 0 : train_data.shape[1] - 2]
    train_y = train_data.iloc[:, train_data.shape[1] - 1 : train_data.shape[1]]
    test_X = test_data.iloc[:, 0 : test_data.shape[1] - 2]
    test_y = test_data.iloc[:, test_data.shape[1] - 1 : test_data.shape[1]]

    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    try_c = [0, 0.00001, 0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,
             1.0, 1.3, 1.8, 2.0]
    test = []
    train = []
    for c in try_c:
        model = SVM(c = c, sigma = 1000, d = 2, b = 0.1, type = 'linear')
        model.train(train_X, train_y)
        pred_test = model.predict(X = test_X)
        test.append(accuracy_score(test_y, pred_test))
        pred_train = model.predict(X = train_X)
        train.append(accuracy_score(train_y, pred_train))
        print(c)
        print(classification_report(test_y, pred_test))
    plt.plot(try_c, test, label = 'testing set')
    plt.plot(try_c, train, label = 'training set')
    plt.ylabel('accuracy')
    plt.xlabel('c')
    plt.title(f'effect on sigma ={model.sigma}')
    plt.legend()
    plt.show()