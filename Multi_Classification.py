import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import Logistic_Regression as LR
def load_data(path):
    data = loadmat(path)
    X = data["X"]
    y = data["y"]
    return X,y
#X, y = load_data('E:\chen yuanyao\Machine Learning\ex1-ex8-matlab\ex3\ex3data1.mat')
#print(np.unique(y))
#print(X,y)
#print(X.shape,y.shape)
#其中有5000个训练样本，每个样本是20*20像素的数字的灰度图像。每个像素代表一个浮点数，表示该位置的灰度强度。
#20×20的像素网格被展开成一个400维的向量。在我们的数据矩阵X中，每一个样本都变成了一行，这给了我们一个5000×400矩阵X，每一行都是一个手写数字图像的训练样本。
def plot_an_image(X):
    random = np.random.randint(0,5000)
    image = X[random,:]
    fig, ax = plt.subplots(figsize = (1, 1))
    ax.matshow(image.reshape(20,20),cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("this is picture{}".format(y[random]))
#plot_an_image(X)

def plot_100_images(X):
    """ 
    随机画100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)
    
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)),
                                   cmap='gray_r')
    plt.xticks([])
    plt.yticks([])        
    plt.show()
#plot_100_images(X)

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def Reg_Cost(theta,X,y,l=1):
    _theta = theta[1:]
    reg = (1 / (2 *len(X))) +(_theta @ _theta)
    return LR.cost_function(theta,X,y) + reg

def Reg_Gradient(theta,X,y, l=1):
    reg = (1/len(X)) * theta
    reg[0] = 0
    return LR.Gradient_Descent(theta,X,y) + reg

def one_vs_all (X,y,l,k):
    """
    l: 正则化参数lamda
    k: 需要进行分类的种数
    通过建立K个逻辑回归分类器，这个函数可以返回这K个分类器对应的θ值。因此返回的应该是一个(K,(n+1))的矩阵
    """
    final_theta = np.zeros((k,X.shape[1]))
    #因为默认的i值从0开始，但为了符合一般观念——即第一个分类器对应第一次循环，所以改范围
    for i in range(1, k + 1):
        theta = np.zeros(X.shape[1])
        #不懂，此处做标记
        y_i = np.array([1 if label == i else 0 for label in y])
        print(y_i)
        """
        此函数为scipy库中的minimize函数。具体信息参考python内部的源代码解释。这里只简单介绍各参数的用法（并且仅为个人在看完注释之后的理解，不一定对）：
        minimize：对目标函数求根据一个或多个参数求最小值
        fun：对象函数，即需要求最小值的函数，这里用的是自己写的正则化代价函数
        x0：ndarray, shape (n,)。这是对于目标函数参数的初始化猜测，我认为即随机初始化。所以theta是一个0矩阵
        args：tupple（元组）（optional）。调用目标函数需要的额外的参数
        method：选择solver。选择一个解题器。这里用的是TNC：使用Newton Conjugate-Gradient算法。下面是method的列表
            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
        jac：{callable,  '2-point', '3-point', 'cs', bool}, optional。用于计算梯度向量的函数。只在以下method使用时可以使用 CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr. 注意其中的callable的意思是函数，填入的该函数必须返回一个
        梯度向量
        options：dict, optional。 A dictionary of solver options. All methods accept the following。
            generic options:
                maxiter : int
                    Maximum number of iterations to perform. Depending on the
                    method each iteration may use several function evaluations.
                disp : bool
                    Set to True to print convergence messages.
        return：res : OptimizeResult：返回最优化的结果该结果有以下性质：
        ``x`` the solution array
        ``success`` a Boolean flag indicating if the optimizer exited successfully
        ``message`` describes the cause of the termination.
        """
        res = minimize(fun = Reg_Cost, x0 = theta, args = (X, y_i,l), method = "TNC", jac = Reg_Gradient, options = {'disp' : True})
        final_theta[i - 1 : ] = res.x

    return final_theta

def predict_all(X, final_theta):
    # 计算各类的可能性
    possibility = sigmoid_function(X @ final_theta.T)
    #返回最大的索引（列方向）
    max_possibility = np.argmax(possibility, axis = 1)
    #使其符合常识
    max_possibility += 1
    return max_possibility

raw_X, raw_y = load_data('E:\chen yuanyao\Machine Learning\ex1-ex8-matlab\ex3\ex3data1.mat')
X = np.insert(raw_X, 0, 1, axis=1) # (5000, 401)
y = raw_y.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）

print(y)
all_theta = one_vs_all(X, y, 1, 10)
all_theta  # 每一行是一个分类器的一组参数
y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))