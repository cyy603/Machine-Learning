import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import os
import cv2
import glob

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
    X, num0 = read_image(path + '/0')
    y = np.zeros((num0, 1))
    for i in range(1, 5):
        temp_X, num = read_image(path + '/' + str(i))
        temp_y = np.zeros((num, 1)) + i
        X = np.row_stack((X, temp_X))
        y = np.row_stack((y, temp_y))
    return X, y

if __name__ == '__main__':
    train_X, train_y = read_data('E:\\chen yuanyao\\Machine Learning\\SDM\\Week4\\Homework4\\Digits\\train')
    test_X, test_y = read_data('E:\\chen yuanyao\\Machine Learning\\SDM\\Week4\\Homework4\\Digits\\test')
    mlp = MLPClassifier(hidden_layer_sizes = (100, 80 , 60, 40, 20), max_iter = 3000, alpha = 10)
    mlp.fit(train_X, train_y.ravel())
    predict = mlp.predict(test_X)
    numbers = ['0', '1', '2', '3', '4']
    print(classification_report(test_y, predict, target_names = numbers))
