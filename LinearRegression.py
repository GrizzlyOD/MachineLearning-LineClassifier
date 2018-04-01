from sklearn.datasets import load_boston
boston = load_boston()

#print(boston.data.shape)
#(506, 13)
#print(boston.DESCR)
#Missing Attribute Values: None

from sklearn.model_selection import train_test_split
import numpy as np
x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)

#分析回归目标值的差异
# print("The max target value is ",np.max(boston.target))
# print("The min target value is ",np.min(boston.target))
# print("The average target value is ",np.average(boston.target))
# The max target value is  50.0
# The min target value is  5.0
# The average target value is  22.532806324110677
#print(boston.target) target --->y data ---->x

#标准化处理
#由于target的值差异较大，需要对特征以及目标值进行标准化处理
from sklearn.preprocessing import StandardScaler

#初始化特征值和目标值的标准化器
ss_x = StandardScaler() #for data
ss_y = StandardScaler() #for target
import pandas as pd
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train =ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))
# y_train =ss_y.fit_transform(pd.colum_or_1d(y_train))
#
#
# y_test = ss_y.transform(pd.colume_or_1d(y_test))

#数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_predict = lr.predict(x_test)
#线性回归

from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=5)     #最大迭代5
sgd.fit(x_train,y_train.ravel())   #将y降维为1维
sgd_y_predict = sgd.predict(x_test)
#随机梯度下降

print("The value of default measurement of LinearRegression is ",lr.score(x_test,y_test))
from sklearn.metrics  import r2_score , mean_absolute_error , mean_squared_error
print("The value of R-squared of LinearRegression is ",r2_score(y_test,lr_y_predict))
print("The mean of squared error of LinearRegression is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print("The mean of absolute error of LinearRegression is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print("_______________________________________________________________")
print("The value of default measurement of SGDRegression is ",lr.score(x_test,y_test))
print("The value of R-squared of SGDRegression is ",r2_score(y_test,sgd_y_predict))
print("The mean of squared error of SGDRegression is ",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict)))
print("The mean of absolute error of SGDRegression is ",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict)))