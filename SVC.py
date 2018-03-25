
#数据预处理
#导入手写体数字加载器
from sklearn.datasets import load_digits
#通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits()
#查看数据规模和特征维度
print(digits.data.shape)

'''用交叉验证分割样本为训练样本和测试样本'''
from sklearn.model_selection import train_test_split
#随机选取75%作为训练样本，其余25%为测试样本
x_train,x_test,y_train,y_test= train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
#查看训练与测试的数据规模
print(y_train.shape,y_test.shape)

'''使用支持向量机（分类）对手写数字图像进行识别'''
from sklearn.preprocessing import StandardScaler
#从支持向量机中导入线性的支持向量分类器
from sklearn.svm import LinearSVC
ss= StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

#初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
#进行模型训练
lsvc.fit(x_train,y_train)
#利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中
y_predict = lsvc.predict(x_test)
#精确度评分
print('The Accuracy of Linear SVC is ',lsvc.score(x_test,y_test))
#用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names= digits.target_names.astype(str)))
#target_names 接收的是str类型 而digits.target_names返回的是numpy.arange(10) 要做类型转换
#对多分类样本一般将样本看成阴（负）样本，创造10个二分类任务