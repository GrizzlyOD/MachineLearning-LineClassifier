import pandas as pd
import numpy as np

#创建特征列表
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
                'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
#使用pandas.read_csv函数从互联网读取制定的数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names = column_names)

#将缺失值？替换为标准缺失值表示
data = data.replace(to_replace='?',value=np.nan)
#丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')
#输出data的数据和维度
print(data.shape)
#以上是对数据进行预处理

#sklearn.cross_valiation移除在0.20中 替换为 sklearn.model_selection
from sklearn.model_selection import train_test_split
#随机采样25%的数据用于测试，剩下的75%用于构建训练集合
x_train,x_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size = 0.25,random_state=33)
#查验训练样本的数量和类别分布
print(y_train.value_counts())
#查询测试样本的数量和类别分布
print(y_test.value_counts())

#使用线性分类模型从事良/恶性肿瘤预测任务
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import  SGDClassifier

#标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值而主导
ss=StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#初始化LogisticRegression与SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier(max_iter=5)           #in 0.19 dedault to max_iter=5 and tol=None
#调用LogisticsRegression中的fit函数/模块用来训练模型参数
lr.fit(x_train,y_train)
#使用训练好的模型lr对x_test进行预测，结果储存在变量lr_y_predict中
lr_y_predict = lr.predict(x_test)
#调用SGDClassifier里的fit函数/模块用来训练模型函数
sgdc.fit(x_train,y_train)
#使用训练好的模型sgdc对x_test进行预测，结果储存在sgdc_y_predict中
sgdc_y_predict = sgdc.predict(x_test)

#positive->阳性(恶性)     negative->阴性(良性)
#准确度Accuracy = (true positive + true negative)/(true positive + false positive + true negative + false negative)
#召回率Recall = (true positive)/(true positive + false negative)
#精确度Precision = (true positive)/(true positive + false positive)
#f1 = 2/((1/precision)+(1/recall))

#使用线性分类模型对性能进行分析
from sklearn.metrics import classification_report

#使用逻辑斯蒂回归函数自带的评分函数score获得模型在测试集上的精确性结果
print('Accuracy of LR Classifier:',lr.score(x_test,y_test))
#利用classification_report模块获得LogisticsRegression其他三个指标的结果
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))


#使用随机梯度下降模型自带的评分函数score获得模型在测试集上的精确性结果
print('Accuracy of SGD Classifier:',sgdc.score(x_test,y_test))
#利用classification_report模块获得SGDClassifier其他三个指标的结果
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))

#benign = 良性  malignant = 恶性