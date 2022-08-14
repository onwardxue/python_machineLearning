# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:55 上午
# @Author : Bin Bin Xue
# @File : sentiment_analysis
# @Project : python_machineLearning

'''
第八章 用机器学习进行情感分析
    8.1 为文本处理预备好IMDb电影评论数据

    8.2 词袋模型介绍
        1_词袋：将文本转换为数字型的特征向量
            -为整个文档集创建唯一的令牌（如单词）词汇表
            -为每个文档创建一个特征向量，包含每个词在特定文档出现的频率（稀疏向量）
        2_构建词袋模型
        3_词频-逆文档频率-删除各文档都频繁出现的词汇
        4_清洗文本数据-去除无意义的符号
        5_把文档处理成令牌-段落/句子分成单词，去除停用词

    8.3 训练用于文档分类的逻辑回归模型
        如下代码
    8.4 处理更大的数据集-在线算法和核外学习
        在线学习-小批量读取数据，分批进行模型训练
    8.5 用潜在狄利克雷分类实现主题建模
        LDA实现无监督分类（根据电影评论的高频词分成不同的主题）
    8.6 本章小结
        nlp基础：（电影评论分类）
            将电影评论分解成单词，单词作为特征，根据评论中单词出现的频数预测其属于正/负评论。
            使用lda还能将其分成多种类别（主题），根据不同类别的单词重要性排序，可以判断其属于
            哪种类型的电影。
'''
import numpy as np
import pandas as pd
import pyprind as pyprind
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import nltk
import os
import gzip
# 1.1_读取表格数据
df = pd.read_csv('movie_data.csv',encoding='utf-8')
print(df.head(3))
print(df.shape)
# 1.2_模拟构建词袋模型
def bag_words():
    # 根据单词在各文件中出现的频率构建词袋模型
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
    bag = count.fit_transform(docs)
    # 输出词汇表的内容（根据ascall码排序，每个单词映射到单词字典中，一个下标对应一个单词）
    print(count.vocabulary_)
    # 显示创建的特征向量（每个向量由单词表所含单词的频数组成）
    print(bag.toarray())
    return docs

# 1.3_词频-逆文档频率，去掉多个文档中出现较为频繁的词汇（如is/the，这些词一般不具有任何判断信息）
def to_tfidf():
    docs = bag_words()
    tfidf = TfidfTransformer(use_idf=True,
                             norm='l2',
                             smooth_idf=True)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# 1.4_清洗文本数据（删除不需要的字符，如标点符号）
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)

# 1.5_把文档处理成令牌（句子划分成单词）
# 按空格划分
def tokenizer(text):
    return text.split()
# tokenizer('runners like running and thus they run')

# 用波特词干提取算法提取
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# tokenizer_porter('runners like running and thus they run')

# 1.6_移除'停用词'（is,has,and等，降低这些频繁出现的词汇的权重）
# 下载127个停用词组成的数据包,生成停用词转换器，并用于该语句中
nltk.download('stopwords')
# 英文停用词库转换器
stop = stopwords.words('english')
# [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
#      if w not in stop]

# 1.7_训练一个逻辑回归分类器（将评论分为正面和负面；训测比为25000：25000）
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
# 词频转换器
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
# 网格搜索参数
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
# 管道封装词频转换器和逻辑回归分类器
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])
# 网格搜索，指标为accuracy，5折,n_jobs=-1可以加快搜索速度
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)
# 显示最佳参数、训练集上的准确率
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
# 预测测试集，并显示测试集上的准确率
# 结果表明，模型能以90%的预测率预测电影评论是正还是负
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# 2.0 处理更大的数据集-使用在线算法和核外学习（计算速度的更快）
# 英文停词库
stop = stopwords.words('english')
# 文本清洗器
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# 生成器函数（每次读取并返回一个文档）
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# 验证生成器
# next(stream_docs(path='movie_data.csv'))

# 读入文档流并通过参数size返回指定数量的文档
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# 向量化
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)
# 初始化逻辑回归分类器
clf = SGDClassifier(loss='log', random_state=1)
# 读取数据
doc_stream = stream_docs(path='movie_data.csv')
# pyprind用于估计机器学习算法进度（定义45个批次，每个批次1000个文档；剩下5000个用于测试）
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    # 每轮读取1000个进行训练，模拟在线学习（小批次）
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# 读取剩余5000个为测试集，输出预测情况
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
# 继续用剩下的5000个更新模型，方便预测其它数据集
clf = clf.partial_fit(X_test, y_test)

# 3.0 用潜在狄利克雷分配实现主题建模 - 根据电影评论划分不同主题（LDA)
# 重新读取数据
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
# 创建词袋，作为LDA的输入
count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)
# 训练LDA，设置划分个数为10个，batch表示在每次迭代中都使用所有数据
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)
# 输出lda维度，看是否是10个
print(lda.components_.shape)
# 获取各主题最重要的5个特征
n_top_words = 5
feature_names = count.get_feature_names()
# 输出各主题最重要的5个特征
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort() \
                        [:-n_top_words - 1:-1]]))

# Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
#
# 1. Generally bad movies (not really a topic category)
# 2. Movies about families
# 3. War movies
# 4. Art movies
# 5. Crime movies
# 6. Horror movies
# 7. Comedies
# 8. Movies somehow related to TV shows
# 9. Movies based on books
# 10. Action movies

# 输出第6个主题（恐怖类）的前三条评论，看是否符合
horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')