# -*- coding:utf-8 -*-
# @Time : 2022/8/11 10:21 上午
# @Author : Bin Bin Xue
# @File : cluster&unlabeled_data
# @Project : python_machineLearning

'''
第11章 用聚类分析处理无标签数据
        无监督学习技术-自然分组（根据对象相似性）
        - 3种技术
            - 基于原型（K均值）、层次化、基于密度

    11.1 k-均值进行相似性分组
        （1）基于原型指每个类都有一个原型（质心）表示，该原型为类中心，是最具代表性的点(假设集群成球型)
        （2）应用范围最广，音乐/电影/购物群体..分组，从而构建推荐系统
        （3）缺点是必须提前指定K值，属于先验方法
        （4）使用K均值++更好(更好地设置集群质心，init='k-means++')
        （5）硬聚类：一个样本分配给一个集群；软聚类：一个样本分配给一个或多个集群
        （6）评定最优集群数K：肘部方法（绘制失真图）
        （7）集群种样本分组的紧密程度：轮廓分析（轮廓系数-集群内聚度+集群分离度）
                轮廓系数图
                - 比较不同集群的形状大小
                - 看下边是否接近于0，为接近于0表示良好
                - 虚线为平均轮廓系数

    11.2 把集群组织成层次树
        （1）层次聚类：凝聚+分裂（优点是可解释性好，能绘制树图）
        （2）skl-AgglomerativeClustering（能返回所属集群标签、集群数量）
        （3） scipy-linkage

    11.3 通过DBSCAN定位高密度区域
        （1）密度：指定半径范围内的样本数据点数
        （2）核心点：指定数量的相邻点落在以该点为圆心的指定半径内
        （3）边界点：指定半径内，但相邻点数量达不到指定数量的
        （4）优点：不呈球型分布，能完成任意形状的聚类；能够去除异常值（不用每个点都分配到集群中）
        （5）缺点：无法适用特征数目太多；数据密度差别较大时，难以找到优的MinPts/e组合

    11.4 本章小结
        介绍了三种聚类：
        （1）k-均值（预先定义集群数，无监督，聚集成球形）
        （2）凝聚层次聚类（不用预先定义成集群数，可视化方便，可解释性高）
        （3）DBSCAN（处理非球状集群）
        补充：
        （1）更高级和聚类-基于图形的聚类
        （2）维数过高会极大影响聚类，因此要提前降维
        （3）最重要的是距离度量和领域知识，才是超参数和算法


'''
import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# 0_创建简单的二维数据
from sklearn.metrics import silhouette_samples

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
# plt.savefig('images/11_01.png', dpi=300)
plt.show()
# 1_k均值集群分类
def kmeans():
    km = KMeans(n_clusters=3,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)

    y_km = km.fit_predict(X)
    # 绘制k均值在数据集中发现的集群和质心点
    plt.scatter(X[y_km == 0, 0],
                X[y_km == 0, 1],
                s=50, c='lightgreen',
                marker='s', edgecolor='black',
                label='Cluster 1')
    plt.scatter(X[y_km == 1, 0],
                X[y_km == 1, 1],
                s=50, c='orange',
                marker='o', edgecolor='black',
                label='Cluster 2')
    plt.scatter(X[y_km == 2, 0],
                X[y_km == 2, 1],
                s=50, c='lightblue',
                marker='v', edgecolor='black',
                label='Cluster 3')
    plt.scatter(km.cluster_centers_[:, 0],
                km.cluster_centers_[:, 1],
                s=250, marker='*',
                c='red', edgecolor='black',
                label='Centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    # plt.savefig('images/11_02.png', dpi=300)
    plt.show()
    # 输出SSE
    print('Distortion: %.2f' % km.inertia_)

kmeans()

# 2_绘制肘部图找最优集群数
def elbow_plot():
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    # plt.savefig('images/11_03.png', dpi=300)
    plt.show()
elbow_plot()
# 3_通过轮廓图量化聚类质量（轮廓分析）
def silhouette_analysis():
    km = KMeans(n_clusters=3,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    # plt.savefig('images/11_04.png', dpi=300)
    plt.show()
silhouette_analysis()
# Comparison to "bad" clustering:
# 4_k均值分类结果不好的情况
def bad_clustering():
    # 绘制聚类图，看聚类情况，质心位置
    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    plt.scatter(X[y_km == 0, 0],
                X[y_km == 0, 1],
                s=50,
                c='lightgreen',
                edgecolor='black',
                marker='s',
                label='Cluster 1')
    plt.scatter(X[y_km == 1, 0],
                X[y_km == 1, 1],
                s=50,
                c='orange',
                edgecolor='black',
                marker='o',
                label='Cluster 2')

    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', label='Centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.savefig('images/11_05.png', dpi=300)
    plt.show()
    # 绘制轮廓图，看不同集群的长度和宽度相似度
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    # plt.savefig('images/11_06.png', dpi=300)
    plt.show()
bad_clustering()
# # Organizing clusters as a hierarchical tree

# ## Grouping clusters in bottom-up fashion
# 5_层次聚类
def hierarchical_clustering():
    # 产生随机样本数据
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

    X = np.random.random_sample([5, 3]) * 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df)

    # ## Performing hierarchical clustering on a distance matrix
    row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                            columns=labels,
                            index=labels)
    print(row_dist)

    # 1. incorrect approach: Squareform distance matrix
    row_clusters = linkage(row_dist, method='complete', metric='euclidean')
    pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2',
                          'distance', 'no. of items in clust.'],
                 index=['cluster %d' % (i + 1)
                        for i in range(row_clusters.shape[0])])

    # 2. correct approach: Condensed distance matrix
    row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2',
                          'distance', 'no. of items in clust.'],
                 index=['cluster %d' % (i + 1)
                        for i in range(row_clusters.shape[0])])

    # 3. correct approach: Input matrix
    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    pd.DataFrame(row_clusters,
                 columns=['row label 1', 'row label 2',
                          'distance', 'no. of items in clust.'],
                 index=['cluster %d' % (i + 1)
                        for i in range(row_clusters.shape[0])])

    # make dendrogram black (part 1/2)
    # set_link_color_palette(['black'])
    row_dendr = dendrogram(row_clusters,
                           labels=labels,
                           # make dendrogram black (part 2/2)
                           # color_threshold=np.inf
                           )
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    # plt.savefig('images/11_11.png', dpi=300,
    #            bbox_inches='tight')
    plt.show()

    # ## Attaching dendrograms to a heat map
    # plot row dendrogram
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

    # note: for matplotlib < v1.5.1, please use orientation='right'
    row_dendr = dendrogram(row_clusters, orientation='left')

    # reorder data with respect to clustering
    df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

    axd.set_xticks([])
    axd.set_yticks([])

    # remove axes spines from dendrogram
    for i in axd.spines.values():
        i.set_visible(False)

    # plot heatmap
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))

    # plt.savefig('images/11_12.png', dpi=300)
    plt.show()

    # 使用skl的agglomerative生成层次聚类，输出各样本分配的集群标签
    ac = AgglomerativeClustering(n_clusters=3,
                                 affinity='euclidean',
                                 linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)

    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='complete')
    labels = ac.fit_predict(X)
    print('Cluster labels: %s' % labels)
hierarchical_clustering()

# # Locating regions of high density via DBSCAN
# 6_DBSCAN
def DBSCAN_compare():
    # 创建新的半月形结构数据集，比较K均值、层次聚类和DBSCAN
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.tight_layout()
    # plt.savefig('images/11_14.png', dpi=300)
    plt.show()

    # K-means and hierarchical clustering:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                edgecolor='black',
                c='lightblue', marker='o', s=40, label='cluster 1')
    ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                edgecolor='black',
                c='red', marker='s', s=40, label='cluster 2')
    ax1.set_title('K-means clustering')

    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity='euclidean',
                                 linkage='complete')
    y_ac = ac.fit_predict(X)
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
                edgecolor='black',
                marker='o', s=40, label='Cluster 1')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red',
                edgecolor='black',
                marker='s', s=40, label='Cluster 2')
    ax2.set_title('Agglomerative clustering')

    plt.legend()
    plt.tight_layout()
    # plt.savefig('images/11_15.png', dpi=300)
    plt.show()

    # Density-based clustering:
    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
                c='lightblue', marker='o', s=40,
                edgecolor='black',
                label='Cluster 1')
    plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
                c='red', marker='s', s=40,
                edgecolor='black',
                label='Cluster 2')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('images/11_16.png', dpi=300)
    plt.show()

DBSCAN_compare()
