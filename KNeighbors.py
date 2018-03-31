from sklearn.datasets import load_iris
#使用加载器读取数据并存储在变量iris中
iris = load_iris()

#查看数据
#print(iris.data.shape)
#(150,4)

#print(iris.DESCR)
#查看数据的描述

#交叉检验，数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

#导入标准化模块和kneighborsclassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss = StandardScaler()
x_train =ss.fit_transform(x_train)
x_test = ss.transform(x_test)

knc = KNeighborsClassifier()
knc.fit(x_train,y_train)
y_predict = knc.predict(x_test)

# print(y_predict)
# print(y_test)
print("The accuracy of KNeighbors Classifier is: ",knc.score(x_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))