# -*- coding:utf-8 -*-
# @Time : 2022/8/12 10:30 上午
# @Author : Bin Bin Xue
# @File : utils
# @Project : python_machineLearning

'''
    工具类
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1_绘制决策边界
# 不同类别的点标记符号不同，分割区域颜色不同，测试集数据用圆圈框出来
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    )

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

# 2_引入鸢尾花数据并进行预处理（查看标签种类，数据划分，特征标准化）
def data_preprocessing(X,y):

    # 2_查看标签种类
    print('Class labels:', np.unique(y))

    # Splitting data into 70% training and 30% test data:
    # 3_数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)


    print('Labels count in y:', np.bincount(y))
    print('Labels count in y_train:', np.bincount(y_train))
    print('Labels count in y_test:', np.bincount(y_test))

    # Standardizing the features:
    # 4_X特征标准化
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std,X_test_std,y_train,y_test

# 3_整合训练和测试数据（用于绘制决策边界）
def combained_train_test(X_train_std,X_test_std,y_train,y_test):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    return X_combined_std,y_combined
