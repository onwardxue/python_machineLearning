# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:21 上午
# @Author : Bin Bin Xue
# @File : machine_learning_basic
# @Project : python_machineLearning

'''
第一章 赋予机器从数据中学习的能力
    1.1 构建能把数据转换为知识的智能机器
        人工智能>'机器学习'

    1.2 三种不同类型的机器学习
        监督学习（用有标签数据训练，目标是根据新数据预测结果）
            分类（标签为有限个数的离散的无序值）
                -如：判断是否是垃圾邮件
            回归（标签为连续值）
                -如：根据学习时间预测考生成绩
        无监督学习（无标签数据，目标是发现数据中隐藏的结构）
            聚类（寻找子群）
                -如：根据客户数据，发现不同的客户群
            降维（压缩数据）
                -如：去噪，优化性能/节省空间和计算性能/可视化
        强化学习（决策过程与奖励机制，目标是开发一个系统(智能体)能实现更好的交互）
            一般模式（智能体通过与环境的一系列交互来最大化奖励）
                -如：国际象棋

    1.3 基本术语与符号
        （1）符号约定
                X（大写粗体：矩阵/数据集）
                x（小写粗体：一行数据/向量/样本）
                x(i)j（上标i：第i行/个样本；下标j：第j列/个特征）
                    x(i)（第i行/个行向量，jmax维）
                    xj（第j列/个列向量，imax个元素）
                    x(i)j（矩阵中的一个元素，位置为(i,j)）
                y（目标变量/预测标签，含有imax个元素的列向量）
        （2）机器学习术语
                训练样本：表中的行（一行数据是一个样本）
                特征：表中的列，还叫变量/属性/输入/预测因子/协变量
                目标：y，还叫结果/分类标签/输出/响应变量/因变量/真值
                训练：模型拟合（或参数估计）.fit
                损失函数：也叫误差函数（与真实的误差）。损失针对单个数据点/代价针对整个数据集

    1.4 构建机器学习系统的路线图
        （1）预处理（原始数据变成模型输入数据）
        （2）训练和选择预测模型（没有完全通用的模型，交叉验证和模型调优）
        （3）模型评估（使用新数据评估模型泛化误差）
        （4）预测新数据（不能对新数据特征缩放或降维等，否则结果会过于乐观）

    1.5 将Python用于机器学习
        python3.6以上，推荐Anaconda
        其他包：
            Numpy、Scipy、sklearn、Matplotlib、pandas

    1.6 本章小结
        全局了解，细节在后面各章


'''