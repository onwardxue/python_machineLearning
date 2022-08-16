# -*- coding:utf-8 -*-
# @Time : 2022/8/11 10:30 上午
# @Author : Bin Bin Xue
# @File : TensorFlow&parallel_training
# @Project : python_machineLearning

'''
第13章 用TensorFlow并行训练神经网络
    0_tensorflow和pytorch大对比（两个热门的深度学习框架）
        -t
            -google开发，更难学，诞生更早，资源更多，更适合生产环境
        -p
            -facebook开发，更容易学，更适合研究和简单开发

    13.1 TensorFlow与模型训练的性能
        (1)Tensor支持GPU，性能更好
        (2)张量：标量(0级)、向量(1级)、矩阵(2级)的推广定义

    13.2 学习TensorFlow第一步
        (1)安装tensorflow
            2.0.0
        (2)创建张量
        (3)对张量形状和数据类型进行操作
        (4)对张量进行数学运算
        (5)张量堆叠..

    13.3 用TensorFlow的Dataset API构建输入流水线
    13.4 在TensorFlow中构建神经网络模型
    13.5 选择多层神经网络的激活函数
    13.6 本章小结

'''
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

a=np.array([1,2,3],dtype=np.int32)
b=[4,5,6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)
