# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:31 上午
# @Author : Bin Bin Xue
# @File : sklearn_classifier
# @Project : python_machineLearning

'''
第三章 sklearn机器学习分类器
    3.1 选择分类算法
        原因：没有能适合所有场景的分类算法，要比较几种不同的算法以选择适合特定问题的最佳模型
        有监督算法训练步骤：
            (1)选择特征并收集训练样本
            (2)选择度量性能的指标
            (3)选择分类器并优化算法
            (4)评估模型性能
            (5)调整算法

    3.2 sklearn第一步-训练感知机
        （1）获取数据df，分出X和y
        （2）按比例划分为训、测集：train_test_split
        （3）标准化X特征：StandardScaler
        （4）引入模型并训练：Perceptron
        （5）预测X_test，并统计其结果与y_test的不同个数
        （6）计算分类准确率：Accuracy=1-(错误数/总数)
                - 用模型.score(x_test,y_test)或accuracy_score(y_test,y_pred)
        （7）绘制感知器模型的决策区：plot_decision_regions
        感知器最大的缺点是：遇到类不可完全线性可分的情况，会出现永不收敛的问题

    3.3 基于逻辑回归的分类概率建模
        特点：
        （1）逻辑回归是一个分类模型，不是回归模型（线性二元分类问题表现较好）
        （2）逻辑回归不仅能给出分类结果，还能给出各类别的预测概率
        过程：
        （1）skl逻辑回归模型多分类设置参数：multi_class='ovr'，排它法；优化默认为'lbfgs'
        （2）某个样本属于特定类的概率：predict_proba
        （3）取概率最大的为样本标签：predict（skl直接返回预测标签）
        （4）通过正则化解决过拟合问题：调节C参数（C为正则化参数入的倒数，所以取值越小，正则化越强，越不会过拟合。10^-4～10^4）
        注意：如果想单独预测样本标签，要将一行数据转为二维数组。
        （如：lr.predict(x_test_std[0,:].reshape(1,-1)）

    3.4 使用支持向量机最大化分类间隔/3.5 用核支持向量机求解非线性问题
        特点：
        （1）感知器的扩展，优化目标是最大化分类间隔
        （2）决策边界又叫'超平面'，最靠近超平面的点为'支持向量'，超平面两侧的支持向量分别组成'正超平面'和'负超平面'，正负之间
             的距离为'间隔'
        （3）参数C控制对分类错误的惩罚，C值越大，错误惩罚越大/间隔宽度越小/越容易过拟合
        （4）核技巧解决非线性分类问题（生成非线性的决策边界）
            -kernel='rbf',gamma=0.10（高斯核；gamma为截止参数，控制决策边界的宽松度，值越大越紧密，越容易过拟合。0.1~100）

    3.6 决策树学习
        特点：
        （1）可解释性最好的分类器
        （2）基尼杂质度量（gini、熵、分类误差；gini最常用，处于两个极端之间）
            -参数criterion='gini'
        （3）skl的决策树不提供手工剪枝的方法，只能提前限制树深
            -参数max_depth=4
        （4）多棵树组成随机森林，属于集成算法之一（不用剪枝，有影响的超参数个数较少）
            -参数n_estimators=25（树的棵树，一般越多越好，但同时也会增大训练时长）
            -n_jobs=2（多核处理，加快训练速度）
            -bootstrap（样本规模，skl默认为原始数据集规模，不用调）
            -d（每轮分裂的特征数，skl默认为根号m，不用调）

    3.7 K-近邻 - 一种惰性学习算法
        特点：
        （1）从训练数据中找到离预测样本最近的k个样本点，多数投票决定标签
        （2）缺点是无法适应大数据量，计算复杂度与数据量呈线性关系，且会占用较大的存储空间；容易过拟合；K值不好确定
        （3）优点是对新数据的预测速度快，因为没有训练过程
        （4）相关参数
            -n_neighbors=5（k值，对结果/拟合程度影响大）
            -metric='minkowski',p=2（距离度量方法，p=2表示欧氏距离，=1表示曼哈顿距离）

    3.8 本章小结
        学习6种分类器的特点、使用和参数设置
            决策树 - 可解释性好，容易过拟合
            逻辑回归 - 可以预测概率
            支持向量机 - 能较好处理非线性问题，但超参数多
            随机森林 - 不用太多参数，且不容易过拟合，应用广泛
            Knn - 无训练过程，但不适用大数据（计算代价大）
'''
import matplotlib
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import plot_decision_regions,data_preprocessing,combained_train_test
# 0_导入鸢尾花数据，并指定特征和标签
def iris_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    return X,y

# ## Training a perceptron via scikit-learn
# 1_用感知器建模预测
def PerceptronPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train_std,X_test_std,y_train,y_test=data_preprocessing(X,y)
    # 2_感知器模型训练
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    # 3_预测
    y_pred = ppn.predict(X_test_std)
    # 4_输出预测错误的个数
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    # 5_输出准确率
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))
    # 6_绘制决策边界图
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    # 调用绘制方法
    plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# 2_用逻辑回归建模预测
def LogisticRegressionPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train_std, X_test_std, y_train, y_test = data_preprocessing(X, y)
    # 2_逻辑回归训练
    lr = LogisticRegression(C=100.0,random_state=1,multi_class='ovr')
    lr.fit(X_train_std,y_train)
    # 3_绘制决策边界
    X_combined_std,y_combined = combained_train_test( X_train_std, X_test_std, y_train, y_test)
    plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    # 自动调整图各元素大小，使其合适
    plt.tight_layout()
    plt.show()
    # 4_预测前三个样本属于各类的条件概率
    print('predict_proba：\n',lr.predict_proba(X_test_std[:3,:]))
    # 5_输出三个样本的预测标签
    # 找最大概率
    print('predict_label：',lr.predict_proba(X_test_std[:3,:]).argmax(axis=1))
    # skl直接输出
    print('predict_label_skl：',lr.predict(X_test_std[:3,:]))
    # 6_控制参数C，处理过拟合问题
    # 7_如果值预测一个样本时，要转一行数据为二维数组
    print('predict_one_sample：',lr.predict(X_test_std[0,:].reshape(1,-1)))

# 3_用支持向量机预测
def SVMPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train_std, X_test_std, y_train, y_test = data_preprocessing(X, y)
    # 2_无核svm训练
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)
    # 3_绘制决策边界
    X_combined_std,y_combined = combained_train_test( X_train_std, X_test_std, y_train, y_test)
    plot_decision_regions(X_combined_std,
                          y_combined,
                          classifier=svm,
                          test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # 4_高斯核svm训练
    svm_kernel = SVC(kernel='rbf', C=1.0, random_state=1,gamma=0.2)
    svm_kernel.fit(X_train_std, y_train)
    # 5_绘制高斯核svm决策边界
    plot_decision_regions(X_combined_std,
                          y_combined,
                          classifier=svm_kernel,
                          test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # 5_高斯核svm训练(gamma=100)
    svm_kernel_r = SVC(kernel='rbf', C=1.0, random_state=1,gamma=100)
    svm_kernel_r.fit(X_train_std, y_train)
    # 5_绘制高斯核svm决策边界
    plot_decision_regions(X_combined_std,
                          y_combined,
                          classifier=svm_kernel_r,
                          test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# 4_用决策树预测
def dtPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train, X_test, y_train, y_test = data_preprocessing(X, y)
    # 2_决策树训练
    tree_model = DecisionTreeClassifier(criterion='gini',
                                        max_depth=4,
                                        random_state=1)
    tree_model.fit(X_train, y_train)
    # 3_绘制决策边界
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined,
                          classifier=tree_model,
                          test_idx=range(105, 150))

    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # 绘制决策树
    # tree.plot_tree(tree_model)
    # plt.show()
    #
    # dot_data = export_graphviz(tree_model,
    #                            filled=True,
    #                            rounded=True,
    #                            class_names=['Setosa',
    #                                         'Versicolor',
    #                                         'Virginica'],
    #                            feature_names=['petal length',
    #                                           'petal width'],
    #                            out_file=None)
    # graph = graph_from_dot_data(dot_data)
    # graph.write_png('tree.png')

# 5_用随机森林预测
def rfPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train, X_test, y_train, y_test = data_preprocessing(X, y)
    # 2_随机森林训练
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train, y_train)
    # 3_绘制决策边界
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined,
                          classifier=forest, test_idx=range(105, 150))

    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# 5_用knn预测
def knnPrd():
    # 1_数据和预处理
    X, y = iris_data()
    X_train_std, X_test_std, y_train, y_test = data_preprocessing(X, y)
    # 2_knn训练
    knn = KNeighborsClassifier(n_neighbors=5,
                               p=2,
                               metric='minkowski')
    knn.fit(X_train_std, y_train)
    # 3_绘制决策边界
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=knn, test_idx=range(105, 150))

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    # 1_感知器分类器
    # PerceptronPrd()
    # 2_逻辑回归分类器
    # LogisticRegressionPrd()
    # 3_svm分类器
    # SVMPrd()
    # 4_决策树分类器
    # dtPrd()
    # 5_随机森林分类器
    # rfPrd()
    # 6_knn分类器
    knnPrd()

if __name__ == '__main__':
    main()
