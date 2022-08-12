# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:44 上午
# @Author : Bin Bin Xue
# @File : reduce_dimension
# @Project : python_machineLearning

'''
第5章 通过降维压缩数据
    5.1 用主成分分析实现无监督降维(PCA)
    （1）PCA可以提取主要特征，可以降维（保留大部分的相关信息）
    （2）原理是寻找高维数据中存在最大方差的方向（长边，域最大的方向）
    （3）pca不使用任何分类标签信息（无监督学习）
    （4）skl的PCA是一个转换器类，首先用训练数据拟合模型，然后用相同模型参数转换训练数据和测试数据
    （5）n_components为主成分数（降维后的维数/特征数,为0时输出各特征主成分值）

    5.2 基于线性判别分析的监督数据压缩(LDA)
    （1）会用到分类标签信息，但在某些情况下（如每类只包含少量样本时），还是pca效果更好
    （2）能很好地分离两个正态分布
    （3）两个假设：假设数据呈正态分布，假设类具有相同的协方差矩阵
    （4）转换器.fit用的是训练集的标签

    5.3 非线性映射的核主成分分析(KPCA)
    （1）非线性可分问题指的是决策边界无法用直线分割（PCA和LDA都是线性可分）
    （2）KPCA专门用于非线性可分问题（原理也是在数据转到高维线性可分，用PCA，再投影回来）


    5.4 本章小结
        pca - 忽略分类标签，最大化沿正交特征轴的方差，投影到低维
        Lda - 考虑分类标签，最大化类的可分性
        KPCA - 非线性，先投影到更高维用pca，再压缩到更低维子空间


'''
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from util import plot_decision_regions

path = 'wine.data'
df_wine = pd.read_csv(path,header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())
# Splitting the data into 70% training and 30% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Standardizing the data.
# 1_主成分分析
def pcaTest():
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    # 数据降维后使用逻辑回归预测
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_pca, y_train)
    # 绘制训练集边界
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    # 绘制测试集边界
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    # 输出主成分
    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    print(pca.explained_variance_ratio_)

def ldaTest():
    # LDA用训练集特征和标签训练转换器
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    # 多分类逻辑回归
    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_lda, y_train)
    # 绘制训练数据结果
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    # 转换测试集特征(降维)
    X_test_lda = lda.transform(X_test_std)
    # 绘制测试数据结果
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

def kpcaTest():
    X, y = make_moons(n_samples=100, random_state=123)
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)

    plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
                color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

def main():
    # pcaTest()
    # ldaTest()
    kpcaTest()

if __name__ == '__main__':
    main()