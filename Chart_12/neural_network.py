# -*- coding:utf-8 -*-
# @Time : 2022/8/11 10:25 上午
# @Author : Bin Bin Xue
# @File : neural_network
# @Project : python_machineLearning

'''
第12章 从零开始实现多层人工神经网络
    12.1 用人工神经网络建立复杂函数模型
        多层神经网络（多层隐藏层-深度）+用正向传播激活神经网络
    12.2 识别手写数字
        - 手写识别数据集：mnist（直接用skl的方法导入，所以数据结果会与书上有些许不同）
        - 多层感知器模型：neuralnet
    12.3 训练人工神经网络
        逻辑代价函数、反向传播
    12.4 关于神经网络的收敛性
        随机梯度下降避免陷入局部最优
    12.5 为什么要做本章实践而不直接调包？
        了解基本概念
    12.6 本章小结
        程序运行出错？可能版本不一致，也不清楚怎么改这个bug
'''
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
# 1_从skl库中导入数据
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
# 2_归一化像素(-1~1)和预处理数据
X = ((X / 255.) - .5) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=123,stratify=y)
# y_train = y_train.reset_index()
# y_test = y_test.reset_index()
# 3_输出尺寸
print('Rows: %d,columns:%d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d,columns:%d' % (X_test.shape[0], X_test.shape[1]))
print(X_train.head(5))
print(X_test.head(5))
print(y_train.head(5))
print(y_test.head(5))
# 4_将向量特征矩阵变换回原来的28x28像素的图形
# fig, ax = plt.subplots(nrows=2, ncols=5, sharex='True', sharey='True')
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train.values == 7][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
# 5_使用多层感知器进行预测
# 5.1_训练集分成训练集和验证集进行评估
nn = NeuralNetMLP(n_hidden=100,
                  l2=0.01,
                  epochs=200,
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)
nn.fit(X_train=X_train[:55000],
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])

# 5.2_查看迭代次数和代价的趋势情况
plt.plot(range(nn.epochs),nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

# 5.3_查看迭代次数和准确率的趋势情况
plt.plot(range(nn.epochs),nn.eval_['train_acc'],label='training')
plt.plot(range(nn.epochs),nn.eval['valid_acc'],label='validation',linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

# 5.4 计算模型在测试数据集上的准确度来测试其泛化性能
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float)/X_test.shape[0])
print('Test accuracy: %.2f%%' % (acc*100))

# 5.5 查看MLP分类错误的图片情况
miscl_img = X_test[y_test!=y_test_pred][:25]
correct_lab = y_test[y_test!=y_test_pred][:25]
miscl_lab = y_test_pred[y_test!=y_test_pred][:25]

fig,ax = plt.subplots(nrows=5,ncols=5,sharex='True',sharey='True')
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' %(i+1, correct_lab[i],miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
