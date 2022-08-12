# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:36 上午
# @Author : Bin Bin Xue
# @File : data_preprocess
# @Project : python_machineLearning

'''
第四章 构建良好的训练数据集-数据预处理
    4.1 处理缺失数据
        重点：
        （1）统计每列缺失值：
            - isnull().sum()
        （2）删除缺失值（删太多容易缺失信息）
            - dropna(axis,how,thresh,subset=['列名'])
        （3）填充缺失值（最常用方法）
            - 均值插补（用每个特征的均值填补特征中的缺失值）
             - SimpleImputer(missing_values=np.nan,strategy='mean')
        （4）sklearn转换器 - 读取数据，进行数据转换（transform）
            （如：est.fit(X_train)-从训练数据中学习参数，est.transform(X_train)-对数据进行转换）
            sklearn估计器 - 读取数据，预测数据（predict）
            （如：前面的分类器）

    4.2 处理类别数据
        重点：
        （1）类别数据
            - 分为序数特征、标称特征（前者可比较大小，如尺码；后者不可比较大小，如颜色）
        （2）序数特征映射
            - 建立特征映射关系 df['特征'].map(特征映射)
                - 反向映射：inv_size_mapping = {v: k for k,v in size_mapping.items()}
        （3）为分类标签编码
            - LabelEncoder
                -.fit_transform（标签列转码）
                -.inverse_transform(反转)
        （4）标称特征(无序)采用独热编码
            - skl的OneHotEncoder
                - drop='first'
            - pandas的get_dummies（更方便，只转变字符串列，其他列不变）
                - drop_first=True 删除冗余列，减少相关性

    4.3 数据集划分为独立的训练数据集和测试集
        - train_test_split
            -参数x，y(训练特征，分类标签)
            -test_size（测试集所占比例）
            -stratify=y（按标签类不同值的比例划分，确保训、测集中的各标签数量比例相同）
        注意：训测比通常为73、64、82。当数据量特别大，如超过100000时，可为91

    4.4 保持相同的特征缩放
        重点：
        （1）大部分算法都需要'特征缩放'，除了决策树和随机森林
        （2）方法包括两个'归一化'和'标准化'
        （3）'归一化'：特征范围归到[0,1]
            - 最小-最大缩放：MinMaxScaler
        （4）'标准化'：标准正态化（均值为0，方差为1）
            - StandardScaler（最常用）
            - RobustScaler（小型数据集含异常值，或容易过拟合的数据集更适合）

    4.5 选择有意义的特征
        （1）通过逻辑回归的权重系数查看特征（penalty=L1时，表示训练的是一个特征选择方法）
            - lr.coef_（权重系数）
            - lr.intercept_（截距）
        （2）序列特征算法(SBS,SKL中没有实现)
            该方法为类似PCA降维类型的算法（能保证在不降低过多精度的基础上，缩小数据量）

    4.6 随机森林评估特征重要性
        （1）模型训练后直接输出各特征重要性
            - rf.feature_importances_
        （2）根据给定的特征重要性阈值缩减特征个数（特征筛选）
            - sfm = SelectFromModel(rf,threshold=0.1,predict=True)
            - x_selected = sfm.transform(x_train)

    4.7 本章小结
        处理缺失值，类别编码，特征选择，特征重要性

'''
import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 0_数据引入
# 数据_1
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
# 查看缺失值
print(df.isnull().sum)
# 查看数据元素值
print(df.values)

# 1.1_缺失值处理_删除
def del_mis():
    # remove rows that contain missing values
    df.dropna(axis=0)
    # remove columns that contain missing values
    df.dropna(axis=1)
    # only drop rows where all columns are NaN
    df.dropna(how='all')
    # drop rows that have fewer than 3 real values
    df.dropna(thresh=4)
    # only drop rows where NaN appear in specific columns (here: 'C')
    df.dropna(subset=['C'])

# 1.2_缺失值处理_填充
def fil_mis():
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)

# 数据_2
df_2 = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df_2.columns = ['color', 'size', 'price', 'classlabel']

# 2.1_类别数据_序数特征映射
def category_ordered_mapping():
    # 建立特征映射表
    size_mapping = {'XL': 3,
                    'L': 2,
                    'M': 1}
    # 映射
    df_2['size'] = df_2['size'].map(size_mapping)
    print(df_2)
    # 反向映射
    inv_size_mapping = {v: k for k, v in size_mapping.items()}
    df_2['size'] = df_2['size'].map(inv_size_mapping)
    print(df_2)

# 2.2_类别数据_标签编码
def label_code():
    # 分类标签编码
    class_le = LabelEncoder()
    y = class_le.fit_transform(df_2['classlabel'].values)
    print(y)
    # 反向映射
    y = class_le.inverse_transform(y)
    print(y)

# 2.3_类别数据_标称特征_独热编码
def Nominal_onehot():
    # 1_skl实现onehot
    X = df_2[['color', 'size', 'price']].values
    color_ohe = OneHotEncoder()
    print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())
    # 2_对特定多列
    X = df_2[['color', 'size', 'price']].values
    c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                                  ('nothing', 'passthrough', [1, 2])])
    print(c_transf.fit_transform(X))
    # 3_pandas实现onehot
    X=pd.get_dummies(df_2[['price', 'color', 'size']])
    print(X)
    # 4_multicollinearity guard in get_dummies（解决多重共线性）
    pd.get_dummies(df_2[['price', 'color', 'size']], drop_first=True)
    # 5_multicollinearity guard for the OneHotEncoder（解决多重共线性）
    color_ohe = OneHotEncoder(categories='auto', drop='first')
    c_transf = ColumnTransformer([('onehot', color_ohe, [0]),
                                  ('nothing', 'passthrough', [1, 2])])
    c_transf.fit_transform(X).astype(float)

# 数据_3
path = 'wine.data'
df_wine = pd.read_csv(path,header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# 3.1_数据集划分
def split_dataset():
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)
    print(X_train.shape[0])
    print(X_test.shape[0])
    return X_train, X_test, y_train, y_test

# 4.1_特征缩放_归一化
def minmax():
    X_train, X_test, y_train, y_test=split_dataset()
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

# 4.2_特征缩放_标准化
def standard():
    X_train, X_test, y_train, y_test=split_dataset()
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

# 5.1_随机森林_特征重要性_按重要性筛选特征
def rf_imp():
    X_train, X_test, y_train, y_test=split_dataset()

    feat_labels = df_wine.columns[1:]
    # 2_输出特征重要性
    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    forest.fit(X_train, y_train)
    # 输出特征重要性
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            align='center')

    plt.xticks(range(X_train.shape[1]),
               feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
    # 2_按指定阈值筛选重要性较强的特征
    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
    X_selected = sfm.transform(X_train)
    print('Number of features that meet this threshold criterion:',
          X_selected.shape[1])

    # Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):

    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

def main():
    # 1_填充缺失值
    # fil_mis()
    # 2_有序类别映射
    # category_ordered_mapping()
    # 3_类别标签编码
    # label_code()
    # 4_标称特征使用onehot编码
    # Nominal_onehot()
    # 5_划分数据集
    # split_dataset()
    # 6_随机森林查看特征重要性
    rf_imp()


if __name__ == '__main__':
    main()