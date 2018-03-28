#朴素叶贝斯新闻分类
from sklearn.datasets import fetch_20newsgroups
#feth_20newsgroups 需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
#验视数据样本
print(len(news.data))


#交叉检验
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

#文本向量转换
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

#导入naive_bayes模型MultinomialNB
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_predict = mnb.predict(x_test)

#性能评估
from sklearn.metrics import classification_report
print('The accuracy of Naive Bayes Classifier is ',mnb.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))
