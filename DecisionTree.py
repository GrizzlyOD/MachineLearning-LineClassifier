import pandas as pd
titanic  = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print(titanic.head())
#pandas独有的dataframe格式（二维数据表）
#print(titanic.info())

#特征的选择，需要一定的背景知识
x = pd.DataFrame(titanic[['pclass','age','sex']]) #!!新建一个dataframe 而不是对眼来的特征提取因为下面要用到fillna并且inplace=True
y = titanic['survived']

#对选择的特征进行探查
#print(x.info())
'''
pclass    1313 non-null object
age       633 non-null float64
sex       1313 non-null object'''
#1）其中age需要补全（共有1313个数据）
#2）pclass和sex是类别型的，需要转化为数值特征，用1/2/3 和0/1代替


#补充age数据，使用平均数后中位数对模型偏差造成最小影响
x['age'].fillna(x['age'].mean(),inplace=True)

#print(x.info())
#交叉检验
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)

#使用特征转换器对pclass和sex转换
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

#特征转换后，发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的保持不变
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
#print(vec.feature_names_)

#同样测试数据也需要转换
x_test = vec.transform(x_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_precidt = dtc.predict(x_test)

from sklearn.metrics import classification_report
print('The accuracy of DecisionTree classifier is :',dtc.score(x_test,y_test))
print(classification_report(y_test,y_precidt,target_names=['died','survived']))