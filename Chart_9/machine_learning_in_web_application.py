# -*- coding:utf-8 -*-
# @Time : 2022/8/11 9:59 上午
# @Author : Bin Bin Xue
# @File : machine_learning_in_web_application
# @Project : python_machineLearning

'''
第九章 将机器学习模型嵌入web应用
    9.0 介绍
        机器学习技术不局限于离线应用和分析，还可以成为web服务的预测引擎。
        web中常用的有价值的机器学习模型包括：垃圾邮件检测、搜索引擎、推荐系统等等
    9.1 序列化拟合的sklearn估计器（当前代码）
        序列化(保存成文件)-pickle.dump
        反序列化(从文件中读取)-pickle.load

    9.2 搭建SQLite数据库存储数据（当前代码）
        SQLite(python自带数据库)
            -conn = sqlite3.connect(连接和创建数据库)
            -...如下代码
            （用sql语句编写）

    9.3 用Flask开发web应用
        Flask框架(python编写)
            1_第一个flask web应用（1_st_flask_app_1目录）
                -app.py(主要代码)
                    -app = Flask(__name__) 初始化flask实例(说明templates在当)
                    -@app.route('/')指定url，触发index函数执行
                    -index直接渲染templates目录下的文件
                    -app.run表示直接执行脚本
                -templates/first_app.html 主页面

            2_表单验证与渲染（2_st_flask_app_1目录）
                -加载wtforms库收集数据
                -app.py
                    -HelloForm类生成表单， TextAreaField方法验证用户输入信息
                    -hello,验证表单信息后转到hello.html(post转发)
                -formhelpers.html
                    - 渲染文本
                -style.css
                    - 修改文本样式，在html中被引用
                -hello.html
                    - 结果页面（显示用户输入的信息）

    9.4 将电影评论分类器转换为web应用
        实现功能：根据用户输入的评论，判断其正/负？
        功能流程：->用户在界面1表单中输入评论
                ->点击提交按钮进行提交
                ->跳转到界面2，显示预测标签/预测概率，和两个按钮（预测结果=正/否）,和一个按钮（提交另一条评论）
                ->用户点击正或负，会跳转到界面3'谢谢反馈'（模型根据用户反馈进行更新）
                ->界面3也有一个按钮，再次提交评论，会跳到界面12
                ->用户点击再次提交评论，会跳回到界面1
                ->此外，还会将用户提交的评论和模型对它的预测存入SQLite数据库中
        1.第一个web预测模型部署（movieclassifier）
            - app.py（主要后端逻辑）
                第一部分（其他函数）
                    - 加载模型（pickle.load()）
                    - 在根目录下新建数据库文件（os.path.join(cur_dir,'reviews.sqlite')）
                    - 模型预测函数（classify，根据评论返回预测结果）
                    - 模型训练函数（train，根据单个评论，partial_fit模型）
                    - 数据存放到数据库（sqlite_entry，评论和预测标签存入数据库）
                第二部分（各页面逻辑）
                    - 类ReviewForm(渲染登陆页面，设置表单输入字符数不少于15)
                    - index（设置开始界面,渲染界面reviewform.html）
                    - result（如果表单验证且为post，就取输入的评论进行预测，返回渲染后的result.html）
                    - feedback（根据用户反馈，更新模型（正确就不跟新，错误就重练），数据存入数据库），返回渲染后的thanks.html
            - templates目录（html模版文件，3个界面）
                -reviewform.html(首页)
                -results.html(结果页面模版)
                -thanks.html(反馈和感谢)
            - static目录（css文件）
            - pkl_objects（两个pickle文件）

    9.5 在公共服务器上部署web应用
        2.持久化更新在线预测模型（在前面的基础上，增加持久化更新方法）
            - update.py
                - 重新读取数据库中的数据训练模型（保证每次都开网页的模型都是基于前面的基础上训练好的模型）
            - 模型定时备份（每次更新都备份一个，保留时间）
                timestr = time.strftime(%Y%m%d-%H%M%S)
                orig_path = os.path.join(cur_dir,'pkl_objects','classifier.pkl')
                backup_path = os.path.join(cur_dir,'pkl_objects','classifier_%s.pkl' %timestr)
                copyfile(orig_path,backup_path)

    9.6 本章小结
        这是非常有用的一章，清楚了如何将预测模型部署到网络上，实现在线预测、在线更新模型（备份）。。

'''

import numpy as np
from nltk.corpus import stopwords
import re
import pyprind as pyprind
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import os
import pickle
from movieclassifier.vectorizer import vect
import sqlite3

path = '../Chart_8/movie_data.csv'

# 生成的文件都放到gener目录下
gener = './movieclassifier/'

# 1_保存模型和停词器
def save_to_pickle(stop,clf):
    dest = os.path.join('movieclassifier','pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'))
    pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'))

# 2_前一章的核外逻辑回归模型进行情感分析
def sentiment_analysis():
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
    doc_stream = stream_docs(path=path)
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

    # 保存数据和模型到指定位置
    save_to_pickle(stop,clf)

# 这一步会生成一个目录-movieclassifier，底下存有stop、clf的pickle文件
# sentiment_analysis()

# 3_反序列化（读取模型进行预测）-执行成功，说明反序列化成功
def read_in_pickle():
    clf = pickle.load(open(os.path.join('./movieclassifier/pkl_objects','classifier.pkl'),'rb'))

    label = {0:'negative',1:'positive'}
    example = ["'I love this movie. It's amazing."]
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

# read_in_pickle()

# 4_创建和连接sqlite数据库
def sqlite_cnn():
    # 连接sqlite3数据库
    conn = sqlite3.connect(gener+'reviews.sqlite')
    # 创见游标
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS review_db')
    # 创建表格项目
    c.execute('CREATE TABLE review_db''(review Text,sentiment INTEGER,date TEXT)')
    # 添加数据
    example1 = 'I love this movie'
    c.execute('INSERT INTO review_db''(review,sentiment,date) VALUES'"(?,?,DATETIME('now'))",(example1,1))

    example2 = 'I disliked this movie'
    c.execute('INSERT INTO review_db''(review,sentiment,date) VALUES'"(?,?,DATETIME('now'))",(example2,0))
    # 保存数据
    conn.commit()
    # 关闭连接
    conn.close()

# sqlite_cnn()

# 5_从数据库中读取数据
def read_data():
    conn = sqlite3.connect(gener+'reviews.sqlite')
    c = conn.cursor()
    c.execute('SELECT * FROM review_db WHERE date')
    results = c.fetchall()
    conn.close()
    print(results)

