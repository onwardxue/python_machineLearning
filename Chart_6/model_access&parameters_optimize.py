# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:48 上午
# @Author : Bin Bin Xue
# @File : model_access&parameters_optimize
# @Project : python_machineLearning

'''
第6章 模型评估和超参数调优的最佳实践
    6.1 用流水线方法简化工作流
        （1）可以用skl的Pipeline类，拟合任意多个转换步骤的模型，并predict到新数据上
        （2）Pipeline封装多个转换器和一个估计器(分类器)，用训练集和训练标签fit，predict测试集
        （3）流程如下：
            pipeline.fit(训练集、训练集标签)->多个转换器依次用上一级结果学习参数(如缩放/降维/学习算法)
            pipeline.predict(测试集)->转换器依次对数据进行转换->分类器预测结果
        （4）最后一个必须是估计器（.predict）

    6.2 K折交叉验证
        （1）为了调整模型复杂度，获得更有普适性的准确率，欠拟合(高偏差)-过拟合(高方差)问题。
        （2）K折交叉验证比holdout（一个数据集只划分成单个训测验）好，用k折就行
        （3）k折为不放回抽样（留一交叉为每轮留一个用于测试集）
        （4）skl方法为cross_val_score()，参数n_jobs能使用多核计算

    6.3 用学习和验证曲线调试算法
        （1）用来观察模型是否处于欠拟合或过拟合问题
            - 欠拟合：训练集、测试集准确率都低（信息不够）
                - 解决：增加模型的参数个数（如收集或增加额外的特征，降低正则化参数）
            - 过拟合：训练集准确率高、测试集准确率低（两者差异大）
                - 解决：增加模型的训练数据（减少模型复杂度、增加正则化参数、减少特征）
        （2）学习曲线：训练集数据规模-Accuracy
        （3）验证曲线：某个模型超参数(如正则项C)-Accuracy
        （4）学习曲线发现问题，验证曲线分析解决方法的有效性（调参效果）

    6.4 网格搜索调优机器学习模型
        （1）机器学习两类参数
                - 训练数据中学习到的参数（如权重）
                - 算法参数/超参数（如正则化参数、树深等）
        （2）网格搜索（选择超参数）
                -目的：根据结果，寻找模型超参数的最优组合（模型调优）
                -GridSearchCV
                    - 参数param_grid(设置各个超参数的取值范围)
                    - 参数refit=True(自动重新在整个训练集上训练最佳模型)
                    - 参数cv=10（使用10折交叉验证）
                    - best_score_(最优性能值)
                    - best_params_(最佳参数值)
                    - best_estimator_(最佳超参数的模型)

        （3）嵌套式交叉验证（选择模型/算法）
                -目的：根据结果，寻找结果最好的模型
                -原理：外部k折交叉（整体数据集）+内部k折交叉（训练集再划分），常用的是5x2交叉验证
                -GridSearchCV(cv=2)+cross_val_score(cv=5)
                    - 属性scoring='accuracy'（可以指定评价指标）

    6.5 不同的性能评估指标
        （1）混淆矩阵
            - 主对角线为分类正确(TP、TN)，负对角线为分类错误(FP、FN)
                - 行标为实际分类，纵标为预测分类
                -       P    N
                   P   TP   FN
                   N   FP   TN
            - confusion_matrix
        （2）精度和召回率
            - err:预测错误(副对角线)/总
            - acc:预测正确(正对角线)/总
            - 精度PRE:预测为正类中预测正确的比例
            - 召回REC/TPR:实际为正类中预测正确的比例
            - F1:PRE和REC的平衡（最常用）
        （3）ROC曲线
            - 特点：纵坐标为tpr，横坐标为fpr
                - 左上角最优（tpr=1,fpr=0）
                - 主对角表示随机
                - AUC为曲线下面积，值越高说明效果越好

        （4）加权宏平均值（多分类）
            - skl不同评分函数中加参数average指定平均的方法即可
                - make_scorer(score_func=precision_score,
                                average='micro')

        （5）类不平衡处理
            - 类不平衡导致的问题
                - 准确率指标失效，要用其他指标（召回率或精度，召回率更重视找出更多正例；精度可能会忽略部分正例）
                - 模型拟合更偏向多数类
            - 处理类不平衡
                - 对少数类的错误预测加大惩罚（skl中，设置class_weight='balanced'）
                - 少数类上采样/多数类下采样（resample）
                    - 平衡到1：1
                - 人工生成少数类（SMOTE）
            - 一个专门用于类不平衡处理的python库-imbalanced-learn

    6.6 本章小结
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve, GridSearchCV, \
    StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from numpy import interp

path = 'wdbc.data'
df = pd.read_csv(path, header=None)

print(df.head())
print(df.shape)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

le.transform(['M', 'B'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)
# 1_封装转换器和估计器（标准化/PCA/逻辑回归）
# Combining transformers and estimators in a pipeline
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))
# 用训练集和标签fit
pipe_lr.fit(X_train, y_train)
# predict测试集，直接得到预测结果
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# 2_k折交叉验证
scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy scores: %s' %scores)
# 平均标准率和标准方差
print('CV accuracy a; %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

# 3_学习曲线
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                                           solver='lbfgs', max_iter=10000))
# train_sizes控制用于生成学习曲线的训练样本数
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
# 计算训练、测试集的均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制训练集结果（均值）
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

# 绘制测试集结果（均值）
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
# 沿y轴上填充颜色
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()

# 4_验证曲线
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train,
                y=y_train,
                param_name='logisticregression__C',
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.show()

# 5_网格搜索，优化超参数
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
# 设置参数范围
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# 设置网格参数（字典型）
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
# 网格搜索
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)
# 用训练集和标签进行训练
gs = gs.fit(X_train, y_train)
# 输出最佳性能和对应的最佳参数
print(gs.best_score_)
print(gs.best_params_)
# 直接用训练好的最佳分类器进行预测
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))


# 6_嵌套式交叉验证，选择模型
# 5x2嵌套交叉验证，比较不同分类器结果
# svm分类器
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))
# 决策树分类器
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

# 7_评估指标_混淆矩阵
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
# 得到混淆矩阵
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()
# 8_评估指标_精度、召回率、F1、auc_roc
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
# skl的默认正例标签为1，自定义标签为0要构建自己的评分器make_score(同时可自定指标，这里设为f1_score)
scorer = make_scorer(f1_score, pos_label=0)
# 参数范围
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
# 要搜索的参数
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]
# 网格搜索
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# 9_评估指标_绘制ROC曲线
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           solver='lbfgs',
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]
# StratifiedKFold按类别分层抽样（保证不同层的类比一致）
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
# 绘制roc曲线
fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                   % (i + 1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('images/06_10.png', dpi=300)
plt.show()

# 10_类不平衡数据_重采样
# 创建一个不平衡的数据集（357：212）
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
# 输出类不平衡预测的准确率，只能达到原来的90%
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100
# 输出正例数量
print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])
# 少数类上采样（如果要对多数类下采样，调换这里的0、1即可）
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
# 输出重采样后正例的数量
print('Number of class 1 examples after:', X_upsampled.shape[0])
# 将重采样数据与原来叠加
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100