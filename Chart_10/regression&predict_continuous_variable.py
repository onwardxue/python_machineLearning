# -*- coding:utf-8 -*-
# @Time : 2022/8/11 10:10 上午
# @Author : Bin Bin Xue
# @File : regression&predict_continuous_variable
# @Project : python_machineLearning

'''
第十章 用回归分析预测连续目标变量
        另一种监督学习-回归分析（预测趋势、关系、值）
    10.1 线性回归简介
        (1)简单线性回归指单变量线性回归-连续型的目标值(响应变量)，多元线性回归为多变量
        (2)回归线-采样点的最佳拟合线
        (3)残差-预测误差，回归线到样本点的垂直线

    10.2 探索住房数据
        (1)查看数据尺寸、数据类型
        (2)可视化查看异常值、变量分布、变量间的线性关系
        (3)相关矩阵查看相关性

    10.3 普通最小二乘线性回归模型的实现
        (1)自定义
        (2)skl实现

    10.4 利用RANSCA拟合鲁棒回归模型
        (1)避免受到异常值的影响（随机抽样一致性）

    10.5 评估线性回归模型的性能
        (1)残差图
        (2)均方误差MSE
        (3)决定系数R平方

    10.6 用正则化方法进行回归
        (1)岭回归
        (2)LASSO
        (3)弹性网络(Elastic Net)

    10.7 多项式回归-将线性回归模型转换为曲线
        （1）多项式回归-非线性回归使用
            -skl实现-polynomialFeatures
        （2）根据公式将数据转为线性关系，再用线性处理

    10.8 随机森林处理非线性关系
        （1）有更好的泛化性能
        （2）对异常值不敏感
        （3）不需要太多的参数优化

    10.9 本章小结
        对线性回归的简单了解

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor,Lasso
# 0_导入数据
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

path = 'housing.data.txt'
df = pd.read_csv(path,header=None,sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())

def scatter_plot():
    # 1_创建散点图矩阵
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

    scatterplotmatrix(df[cols].values, figsize=(10, 8),names=cols, alpha=0.5)
    plt.tight_layout()
    #plt.savefig('images/10_03.png', dpi=300)
    plt.show()

    # 2_绘制相关矩阵的热度图(找到与目标变量相关性较高的变量)
    cm = np.corrcoef(df[cols].values.T)
    hm = heatmap(cm, row_names=cols, column_names=cols)

    # plt.savefig('images/10_04.png', dpi=300)
    plt.show()

scatter_plot()

# 探索两个变量之间的关系
X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()

# 3_线性回归预测
def linear_regression():
    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    lin_regplot(X, y, slr)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    #plt.savefig('images/10_07.png', dpi=300)
    plt.show()
# 辅助函数，根据数据集和预测模型绘制散点图，并添加回归线
def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

linear_regression()

# **Normal Equations** alternative:

# 4_利用RANSAC拟合鲁棒回归模型
# 用于存在多异常值的情况，原理是淘汰异常值，随机抽样
def ransac_model():
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             loss='absolute_loss',
                             residual_threshold=5.0,
                             random_state=0)

    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask],
                c='steelblue', edgecolor='white',
                marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c='limegreen', edgecolor='white',
                marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.legend(loc='upper left')
    #plt.savefig('images/10_08.png', dpi=300)
    plt.show()
    # 计算斜率和截距
    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)

ransac_model()

# 5_评估线性回归的性能
# 重新选择X，y，并划分数据集
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# 评估1_残差图
# 线性回归预测并绘制残差图
# 残差都在预测线上是最完美的
def residual_plot():
    slr = LinearRegression()

    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    plt.scatter(y_train_pred,  y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()

    # plt.savefig('images/10_09.png', dpi=300)
    plt.show()
    return y_train_pred,y_test_pred

y_train_pred,y_test_pred = residual_plot()

# 评估2_均方误差MSE-代价平均值（越小越好）
# 评估3_决定系数-R平方，MSE的标准化（越大越好）
def MSE_R2():
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
MSE_R2()

# 6_正则方法进行回归 - 解决过拟合（岭(ridge)回归、lasso回归、(ElasticNet)弹性网络回归）
def ridge_lasso_elasticNet():
    # 这里使用的是lasso回归，可切换成另外两个
    lasso = Lasso(alpha=0.1)
    # ridge = Ridge(alpha=1.0)
    # elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print(lasso.coef_)

    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

ridge_lasso_elasticNet()

# 7_多项式回归 - 解决变量之间存在非线性关系（如曲线）
def poly_feature():
    # 增加二次多项式
    X = np.array([258.0, 270.0, 294.0,
                  320.0, 342.0, 368.0,
                  396.0, 446.0, 480.0, 586.0])\
                 [:, np.newaxis]

    y = np.array([236.4, 234.4, 252.8,
                  298.6, 314.2, 342.2,
                  360.8, 368.0, 391.2,
                  390.8])

    # 单项式和多项式的比较
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)

    # 拟合多项式
    lr.fit(X, y)
    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)

    # 拟合二次特征
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

    # 绘制结果
    plt.scatter(X, y, label='Training points')
    plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
    plt.xlabel('Explanatory variable')
    plt.ylabel('Predicted or known target values')
    plt.legend(loc='upper left')

    plt.tight_layout()
    #plt.savefig('images/10_11.png', dpi=300)
    plt.show()
    # 计算MSE和R平方（结果表明多项式比线性拟合效果更好）
    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)
    print('Training MSE linear: %.3f, quadratic: %.3f' % (
            mean_squared_error(y, y_lin_pred),
            mean_squared_error(y, y_quad_pred)))
    print('Training R^2 linear: %.3f, quadratic: %.3f' % (
            r2_score(y, y_lin_pred),
            r2_score(y, y_quad_pred)))

poly_feature()

# 为住房数据集中的非线性关系建模
def deal_hosing():
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    regr = LinearRegression()

    # create quadratic features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # fit features
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))


    # plot results
    plt.scatter(X, y, label='Training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='Linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2,
             linestyle=':')

    plt.plot(X_fit, y_quad_fit,
             label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
             color='green',
             lw=2,
             linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.legend(loc='upper right')

    #plt.savefig('images/10_12.png', dpi=300)
    plt.show()
deal_hosing()

# 非线性数据转成线性（按公式），再用线性回归
def transform_linear():
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)

    # fit features
    regr = LinearRegression()
    X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

    # plot results
    plt.scatter(X_log, y_sqrt, label='Training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='Linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2)

    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
    plt.legend(loc='lower left')

    plt.tight_layout()
    #plt.savefig('images/10_13.png', dpi=300)
    plt.show()
transform_linear()

# 8_随机森林处理非线性好关系
# 决策树回归（优点是不用对数据进行转换）
def dt_reg():
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)

    sort_idx = X.flatten().argsort()

    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000s [MEDV]')
    #plt.savefig('images/10_14.png', dpi=300)
    plt.show()
dt_reg()
# 随机森林回归（有更好的泛化性能，对异常值不敏感，没有太多超参数）
def rf_reg():
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1)

    forest = RandomForestRegressor(n_estimators=1000,
                                   criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    # 绘图
    plt.scatter(y_train_pred,
                y_train_pred - y_train,
                c='steelblue',
                edgecolor='white',
                marker='o',
                s=35,
                alpha=0.9,
                label='Training data')
    plt.scatter(y_test_pred,
                y_test_pred - y_test,
                c='limegreen',
                edgecolor='white',
                marker='s',
                s=35,
                alpha=0.9,
                label='Test data')

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.tight_layout()

    #plt.savefig('images/10_15.png', dpi=300)
    plt.show()
rf_reg()