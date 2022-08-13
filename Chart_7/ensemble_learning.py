# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:51 上午
# @Author : Bin Bin Xue
# @File : ensemble_learning
# @Project : python_machineLearning

'''
第七章 组合不同模型的集成学习
    7.1 集成学习
        (1)只要基本分类器性能优于随机猜测，集成分类器的错误率就要比单一基本分类器低

    7.2 多票机制组合分类器（voting）
        (1)skl中已有实现
            - voting=hard(权重一致，少数服从多数)
            - voting=soft(权重不一致，根据概率得到结果)
        (2)结合不同的基分类器（如决策树、knn、lr）
        (3)集成模型调参：

    7.3 bagging-基于bootstrap样本构建集成分类器(bagging)
        (1)'随机森林'就是其中一种典型特例
        (2)又叫bootstrap聚合，会拟合前面不同的分类器
        (3)用的是前一个分类器的随机特征子集，所以会存在很多重复样本

    7.4 通过自适应boosting提高若学习机的性能(adaboost)
        (1)有略微提高
        (2)无放回抽取训练样本的随机子集，并将前一轮被分类错误样本的50%直接加入该子集
        (3)找出使前面两轮训练差异大的样本加入子集
        (4)更重视前一轮被错误分类的样本，提高其权重
        (5)投票机制结合多个基分类器
        (6)容易过拟合

    7.5 本章小结
        voting/bagging/boosting
        拓展：梯度boosting(gbdt)-比xgboost更快（GradientBoostingClassifier和HistGradientBoostingClassifier）
        有时间看看jh
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import product
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# 0_导入数据,进行预处理
iris = datasets.load_iris()
# 只取两个特征和标签
X, y = iris.data[50:, [1, 2]], iris.target[50:]
# 类别标签转换
le = LabelEncoder()
y = le.fit_transform(y)
# 1.2 划分数据集（按类比例）
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=1,
                                                    stratify=y)
# 1_多分类器投票（Voting）
def ensemble_voting():
    # 1.1 建立三个基分类器（lr/dt/knn）
    clf1 = LogisticRegression(penalty='l2',
                              C=0.001,
                              solver='lbfgs',
                              random_state=1)

    clf2 = DecisionTreeClassifier(max_depth=1,
                                  criterion='entropy',
                                  random_state=0)

    clf3 = KNeighborsClassifier(n_neighbors=1,
                                p=2,
                                metric='minkowski')

    # 使用对lr和knn在使用前封装一个标准化
    pipe1 = Pipeline([['sc', StandardScaler()],['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()],['clf', clf3]])
    # 设置分类器标签
    clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

    # 1.2 使用十折交叉得到这三个分类结果（评估标准设为auc）
    print('10-fold cross validation:\n')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

    # 1.3 使用三个模型构成集成模型
    # mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    mv_clf = VotingClassifier(estimators=[('lr',pipe1),('dt',clf2),('knn',pipe3),],voting='soft')
    # 四个模型标签和组成列表
    clf_labels += ['Majority voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]

    # 1.4 用这四个模型进行预测
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

    # 1.5 绘制比较图
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf, label, clr, ls in zip(all_clf,
                                   clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(X_train,
                         y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                         y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,
                 color=clr,
                 linestyle=ls,
                 label='%s (auc = %0.2f)' % (label, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='gray',
             linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    # plt.savefig('images/07_04', dpi=300)
    plt.show()

    # 1.6 绘制决策边界图
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    all_clf = [pipe1, clf2, pipe3, mv_clf]

    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=2, ncols=2,
                            sharex='col',
                            sharey='row',
                            figsize=(7, 5))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            all_clf, clf_labels):
        clf.fit(X_train_std, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                      X_train_std[y_train == 0, 1],
                                      c='blue',
                                      marker='^',
                                      s=50)

        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                      X_train_std[y_train == 1, 1],
                                      c='green',
                                      marker='o',
                                      s=50)

        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(-3.5, -5.,
             s='Sepal width [standardized]',
             ha='center', va='center', fontsize=12)
    plt.text(-12.5, 4.5,
             s='Petal length [standardized]',
             ha='center', va='center',
             fontsize=12, rotation=90)

    # plt.savefig('images/07_05', dpi=300)
    plt.show()

    # 1.7 集成模型调优
    # 查看集成模型可调的超参数
    print(mv_clf.get_params())
    # 设置参数(分类器名__参数)
    params = {'dt__max_depth': [1, 2],
              'lr__clf__C': [0.001, 0.1, 100.0]}
    # 网格搜索得到最佳参数
    grid = GridSearchCV(estimator=mv_clf,
                        param_grid=params,
                        cv=10,
                        scoring='roc_auc',)
    grid.fit(X_train, y_train)
    # 输出每折结果
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_['mean_test_score'][r],
                 grid.cv_results_['std_test_score'][r] / 2.0,
                 grid.cv_results_['params'][r]))
    # 输出最佳参数和accuracy值
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

# 2_bagging
def ensemble_bagging():
    # 数据
    X_train, X_test, y_train, y_test = wine_data()
    # 两个分类器：单决策树、500个决策树组成的bagging分类器
    tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=None,
                                  random_state=1)

    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=1,
                            random_state=1)
    # 单决策树预测结果
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))
    # bagging结果
    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print('Bagging train/test accuracies %.3f/%.3f'
          % (bag_train, bag_test))
    # 绘制比较图
    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=1, ncols=2,
                            sharex='col',
                            sharey='row',
                            figsize=(8, 3))

    for idx, clf, tt in zip([0, 1],
                            [tree, bag],
                            ['Decision tree', 'Bagging']):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0],
                           X_train[y_train == 0, 1],
                           c='blue', marker='^')

        axarr[idx].scatter(X_train[y_train == 1, 0],
                           X_train[y_train == 1, 1],
                           c='green', marker='o')

        axarr[idx].set_title(tt)

    axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)

    plt.tight_layout()
    plt.text(0, -0.2,
             s='Alcohol',
             ha='center',
             va='center',
             fontsize=12,
             transform=axarr[1].transAxes)

    # plt.savefig('images/07_08.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3_boosting（adaboost）
def ensemble_adaboost():
    # 数据
    X_train, X_test, y_train, y_test = wine_data()
    # 两个分类器比较-单个决策树 vs Adaboost(决策树组成的)
    tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=1,
                                  random_state=1)

    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500,
                             learning_rate=0.1,
                             random_state=1)
    # 单个决策树预测结果
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))
    # adaboost预测结果
    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (ada_train, ada_test))
    # 绘制比较图
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))

    for idx, clf, tt in zip([0, 1],
                            [tree, ada],
                            ['Decision tree', 'AdaBoost']):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0],
                           X_train[y_train == 0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train == 1, 0],
                           X_train[y_train == 1, 1],
                           c='green', marker='o')
        axarr[idx].set_title(tt)

    axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)

    plt.tight_layout()
    plt.text(0, -0.2,
             s='Alcohol',
             ha='center',
             va='center',
             fontsize=12,
             transform=axarr[1].transAxes)

    # plt.savefig('images/07_11.png', dpi=300, bbox_inches='tight')
    plt.show()

# 红酒数据
def wine_data():
    path = 'wine.data'
    df_wine = pd.read_csv(path,header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                       'Proline']
    # drop 1 class
    df_wine = df_wine[df_wine['Class label'] != 1]

    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def main():
    # 投票法
    # ensemble_voting()
    # bagging(随机森林)
    # ensemble_bagging()
    # boosting(adaboost)
    ensemble_adaboost()

if __name__ == '__main__':
    main()