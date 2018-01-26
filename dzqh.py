# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:50:40 2018

@author: dell
"""
#导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import seaborn as sns
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)
sns.set(font=myfont.get_name())
#读取数据，测试集，训练集分开
train=pd.read_table('C:/Users/dell/Desktop/dzqh/data/contest_basic_train.tsv')
test=pd.read_table('C:/Users/dell/Desktop/dzqh/data/contest_basic_test.tsv')


#因为Y是目标变量值，所以Y单独一类
train_x=train.drop(['Y'],axis=1)
train_target=train['Y']
#先画几个图
"""train['LOAN_DATE'].value_counts().plot(kind="bar")
plt.title("放款时间")
plt.show()"""
#把本地的划归成1，非本地的划归成0
train.loc[(train['IS_LOCAL']=='本地籍'),'IS_LOCAL']=1
train.loc[(train['IS_LOCAL']=='非本地籍'),'IS_LOCAL']=0
train['IS_LOCAL']=train['IS_LOCAL'].astype(np.float64)
#把教育水平同样的区分，虽然有缺失值却在10%以内，可以把这一项当做零
#也可以说是单独看成一个分类
#print(train['EDU_LEVEL'].unique())#显示教育的种类
#教育水平跨分那么大可以考虑用one-hot matrix来预处理数据
train['EDU_LEVEL'].fillna(0,inplace=True)
train.loc[(train['EDU_LEVEL']=='其他'),'EDU_LEVEL']=0
train.loc[(train['EDU_LEVEL']=='初中'),'EDU_LEVEL']=1
train.loc[(train['EDU_LEVEL']=='专科及以下'),'EDU_LEVEL']=2
train.loc[(train['EDU_LEVEL']=='专科'),'EDU_LEVEL']=3
train.loc[(train['EDU_LEVEL']=='高中'),'EDU_LEVEL']=4
train.loc[(train['EDU_LEVEL']=='本科'),'EDU_LEVEL']=5
train.loc[(train['EDU_LEVEL']=='硕士研究生'),'EDU_LEVEL']=6
train.loc[(train['EDU_LEVEL']=='博士研究生'),'EDU_LEVEL']=7
train.loc[(train['EDU_LEVEL']=='硕士及以上'),'EDU_LEVEL']=6.5
train['EDU_LEVEL']=train['EDU_LEVEL'].astype(np.float64)

##关于婚姻状态的处理，把各种状态转化为数值型
train.loc[(train['MARRY_STATUS']=='已婚'),'MARRY_STATUS']=1
train.loc[(train['MARRY_STATUS']=='未婚'),'MARRY_STATUS']=2
train.loc[(train['MARRY_STATUS']=='离婚'),'MARRY_STATUS']=3
train.loc[(train['MARRY_STATUS']=='离异'),'MARRY_STATUS']=4
train.loc[(train['MARRY_STATUS']=='其他'),'MARRY_STATUS']=5
train.loc[(train['MARRY_STATUS']=='丧偶'),'MARRY_STATUS']=6
train['MARRY_STATUS']=train['MARRY_STATUS'].astype(np.float64)


train['WORK_PROVINCE']=train.fillna(method='bfill')
train['HAS_FUND']=train['HAS_FUND'].fillna(1.0)
#train['LOAN_DATE']=pd.to_datetime(train['LOAN_DATE'])

##要训练的数据
train_xx=train.drop(['Y','REPORT_ID','ID_CARD','LOAN_DATE','AGENT','SALARY'],axis=1)
#print(train_xx.info())

##采用决策树算法
clf=tree.DecisionTreeClassifier()
clf=clf.fit(train_xx,train_target)
print(clf.score(train_xx,train_target))

