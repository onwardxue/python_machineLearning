# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:26 上午
# @Author : Bin Bin Xue
# @File : simple_classify_algorithm
# @Project : python_machineLearning

'''
第二章 训练简单的机器学习分类算法
    2.1 人工神经元-机器学习的早期历史
        1943-提出'神经元'MCP概念
        1957-提出感知器学习规则（最简单的分类器，当不能完全线性可分时会出现永远无法收敛的情况）
        1960-自适应线性神经元（近似逻辑回归）
    2.2 用python实现感知器学习算法
        (1)代码如下（Perceptron）
        (2)该模型根据每个样本调整其两个属性（特征权重和偏置值）（偏置近似正则化项）
        (3)预测结果是根据特征权重与新样本的向量点积+偏置值，为正则返回1，否则为-1

    2.3 自适应线性神经元和学习收敛
        Adaline
            (1)使用批量梯度下降方法（z使用数据整体与权重的点积+偏置计算）
            (2)z上再覆盖一层sigmod函数（类似逻辑回归）
            (3)先计算误差（实际-sigmod(z)）
            (4)权重更新用的是误差点积x，不只是x；偏置更新也是用误差和
            (5)最后用误差平方和得到代价，检测收敛
            (6)学习率和学习次数会起到更大作用
            其他一样
        Adaline_SGD
            (1)使用随机梯度下降（用于大规模机器学习，又叫在线梯度下降，更新权重更频繁）
            (2)在训练中调整学习率（训练数据要求无序、随机，避免重复）

    2.4 本章小结
        介绍了如何用python实现简单的分类器
        （感知机Perceptron,自适应线性神经元Adaline,随机梯度下降AdalineSGD,）
        为下一章用skl包实现更强大的分类器作准备
'''
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1_实现简单的感知器二元分类器（初始化参数为学习率、学习次数（遍历数据次数）、随机种子）
class Perceptron(object):
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # 初始化权重（高斯分布的随机数，随机种子为用户定义）
        # 不用0初始化是因为使用学习率
        # 权重规模为特征数量+1，第0位为偏置单元（阈值/容忍度）
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        # 学习次数，循环读取n遍数据
        for _ in range(self.n_iter):
            errors = 0
            # 逐个样本处理数据
            for xi, target in zip(X, y):
                # 逐个样本处理数据，每次计算预测值与实际差距，结果乘以学习率得到更新率
                # 分类正确时差距为0不更新
                # 分类错误时差距为2或-2（因为类别标签为1和-1）
                update = self.eta * (target - self.predict(xi))
                # 更新特征权重（原特征权重+更新值*该样本特征值）
                self.w_[1:] += update * xi
                # 更新偏置单元（原值+更新值）
                self.w_[0] += update
                # 累加分类错误值
                errors += int(update != 0.0)
                # 记录每次学习的分类错误值
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # 按公式计算表达式z：z = WTx+w0（数据和权重向量点积+偏置得到预测值）
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        # 如果预测值>=0，则返回1，表示正类；否则返回-1，表示负类（根据正负判定）
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 2_用鸢尾花数据测试感知器分类器
def pptest():
    # 读取数据（数据分为三类，1-50为Iris-setosa，51-100为Iris-versicolor，101-150为Iris-virginica）
    path = 'iris.data'
    df = pd.read_csv(path,header=None,encoding='utf-8')
    print(df.tail())
    # 取前100个样本，取两个特征，和最后一个为标签（标签转为1，-1）
    y = df.iloc[0:100,4].values
    y = np.where(y=='Iris-setosa',-1,1)
    x = df.iloc[0:100,[0,2]].values
    # 绘制样本点（两个特征作两个坐标值，点形状为标签值）
    # plt.scatter(x[:50,0],x[:50,1],color='r',marker='o',label='setosa')
    # plt.scatter(x[50:100,0],x[50:100,1],color='b',marker='x',label='versicolor')
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('petal length [cm]')
    # plt.legend(loc='upper left')
    # plt.show()
    # 测试分类器，查看分类错误与迭代次数之间的关系（是否收敛）-错误率越来越低（从第六次开始收敛）
    ppn = Perceptron(eta=0.1,n_iter=10)
    ppn.fit(x,y)
    # plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
    # plt.show()
    # 使用下面的二维数据集决策边界可视化方法
    plot_decision_regions(x,y,classifier=ppn)
    plt.show()


# 3_一个用于二维数据集决策边界可视化的方法
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    )

# 4_自适应线性神经元Adaline（批量梯度下降）
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # 收集的代价（用于检验收敛）
        self.cost_ = []

        for i in range(self.n_iter):
            # 计算整体样本的z值
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression (as we will see later),
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            # 误差=实际-预测
            errors = (y - output)
            # 更新权重（+学习率*（整体数据）点积（误差），不是用与标签的差值了）
            self.w_[1:] += self.eta * X.T.dot(errors)
            # 更新偏置单元（+学习率*误差和）
            self.w_[0] += self.eta * errors.sum()
            # 计算整体代价（误差平方和）
            cost = (errors**2).sum() / 2.0
            # 保存每轮代价
            self.cost_.append(cost)
        return self

    # 一样，根据输入的样本向量计算z值
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 激活函数（）这里没实际用
    def activation(self, X):
        """Compute linear activation"""
        return X

    # 激活函数（z值）
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# 5_随机梯度下降的Adaline
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training examples in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def data_r():
    path = 'iris.data'
    df = pd.read_csv(path, header=None, encoding='utf-8')
    print(df.tail())
    # 取前100个样本，取两个特征，和最后一个为标签（标签转为1，-1）
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    x = df.iloc[0:100, [0, 2]].values
    return x,y

# 6_测试Adaline
def Adaline_test():
    X,y = data_r()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # 不同学习率的对比
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()

# 7_测试Adaline_SGD
def Adaline_SGD_test():
    X,y = data_r()
    # 数据标准化
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # 训练分类器
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)
    # 绘决策边界
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('images/02_15_1.png', dpi=300)
    plt.show()
    # 绘制代价下降情况
    plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')

    plt.tight_layout()
    # plt.savefig('images/02_15_2.png', dpi=300)
    plt.show()

def main():
    # 测试感知器分类器
    # pptest()
    # 测试批量梯度Adaline
    # Adaline_test()
    # 测试随机梯度Adaline_SGD
    Adaline_SGD_test()


if __name__ == '__main__':
    main()