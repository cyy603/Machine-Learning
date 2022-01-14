import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

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
    i = 0
    for filename in os.listdir(path):
        temp_X, num = read_image(path + '/' + filename)
        temp_y = np.zeros((num, 1)) + i
        if i == 0:
            raw_X = temp_X
            raw_y = temp_y
        else:
            raw_X = np.row_stack((raw_X, temp_X))
            raw_y = np.row_stack((raw_y, temp_y))
        i += 1
    return raw_X, raw_y

class Net(nn.Module):
    def __init__(self, features, outputs):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(features, 800), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(800, 400), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(400, 100), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(100, outputs), nn.ReLU())


    def forward(self, X):
        output = self.layer1(X)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        return output

def train(Module, train_X, train_y, test_X, test_y, epoch, lr, fold):
    iter = epoch / 10
    acc_train = 0
    acc_test = 0
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(Module.parameters(), lr = lr)
    running_loss = []
    train_acc = []
    test_acc = []

    for i in range(epoch):
        outputs = Module(train_X)
        optim.zero_grad()
        loss = loss_func(outputs, train_y)
        loss.backward()
        optim.step()
        running_loss.append(loss.item())
        if i % iter == 0:
            train = Module(train_X)
            y_pred = torch.max(train, 1)[1].data.numpy().ravel()
            train_acc.append(accuracy_score(train_y, y_pred))

            test = Module(test_X)
            y_pred = torch.max(test, 1)[1].data.numpy().ravel()
            test_acc.append(accuracy_score(test_y,y_pred))

    show_data(test_acc, train_acc, index = fold)

    train = Module(train_X)
    y_pred = torch.max(train, 1)[1].data.numpy().ravel()
    tempacc_train = accuracy_score(train_y, y_pred)

    test = Module(test_X)
    y_pred = torch.max(test, 1)[1].data.numpy().ravel()
    tempacc_test = accuracy_score(test_y, y_pred)
    print(f'Fold = {fold}\n'
          f'accuracy on training set = {tempacc_train}\n'
          f'accuracy on testing set = {tempacc_test}')
    acc_train = tempacc_train + acc_train
    acc_test = tempacc_test + acc_test

    return Module, acc_train, acc_test, running_loss


def show_data(test_acc, train_acc, index):
    x_hat = np.arange(0, len(test_acc))
    y_hat = test_acc
    x = np.arange(0, len(train_acc))
    y = np.array(train_acc)
    if index < 5:
        loc = i + 3
    else:
        loc = i + 5
    plt.subplot(2, 7, loc)
    plt.plot(x, y, label = 'tr')
    plt.plot(x_hat, y_hat, label = 'te')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title(f'fold = {index}')
    plt.legend()

def predict(Module, test_X, test_y):
    test = Module(test_X)
    pred = torch.max(test, 1)[1].data.numpy().ravel()
    accuracy = accuracy_score(test_y, pred)

    return accuracy

if __name__ == '__main__':
    path = "E:\chen yuanyao\Machine Learning\SDM\Week5\Homework5\data\Cartoon"
    epoch = 500
    lr = 0.001
    i = 0
    acc_test = 0
    acc_train = 0
    loss = []
    tenf = KFold(n_splits = 10, shuffle = True, random_state = 1)
    X, y = read_data(path)
    for train_index, test_index in tenf.split(X):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]

        train_X_t = torch.tensor(train_X).float()
        train_y_t = torch.tensor(train_y.ravel()).long()
        test_X_t = torch.tensor(test_X).float()
        test_y_t= test_y.ravel()

        classifier = Net(X.shape[1], 7)
        classifier_module, temp_acc_train, temp_acc_test, loss = train(Module = classifier, train_X = train_X_t, train_y = train_y_t,
                                                       test_X = test_X_t, test_y = test_y_t, epoch = epoch, lr = lr, fold = i)
        acc_train = acc_train + temp_acc_train
        acc_test = acc_test + temp_acc_test
        i += 1

        plt.subplot(1, 6, 1)
        plt.plot(np.arange(epoch), loss, label = i)

    plt.subplot(1, 6, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.5, hspace=0.5)
    plt.show()
    print(f'Average accuracy on training:{acc_train / 10}\n'
          f'Average accuracy on testing:{acc_test / 10}')
