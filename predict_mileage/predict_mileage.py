#coding: UTF-8
import math
import random
import pandas as pd
import datetime as dt
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, optimizer, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import csv
import category_encoders as ce 

caltegory_list = []

class Model(Chain):
    def __init__(self,in_size):
        super(Model, self).__init__(
            l1=L.Linear(in_size,in_size*20),
            l2=L.Linear(in_size*20,in_size),
            l3=L.Linear(in_size,1),
        )
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)
    

def convert(j,row,tmp,flag):
    flag = False

    if(len(tmp) == 0):
        tmp.append(row[j])
    else:
        for i in range(len(tmp)):
            if(row[j] == tmp[i]):

                flag = True
        if(flag == False):
            tmp.append(row[j])
    row[j] = tmp.index(row[j])
    print(row[j])

x,t = [],[]

df_train = pd.read_csv("train.tsv",delimiter='\t')
df_test = pd.read_csv("test.tsv",delimiter='\t')

list_cols = ['origin']

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')
df_train_ce_onehot = ce_ohe.fit_transform(df_train)
df_test_ce_onehot = ce_ohe.fit_transform(df_test)

del df_train_ce_onehot['id']
del df_test_ce_onehot['id']

del df_train_ce_onehot['car name']
del df_test_ce_onehot['car name']

df_train_ce_onehot = df_train_ce_onehot[df_train_ce_onehot.horsepower != '?']
df_test_ce_onehot =  df_test_ce_onehot[df_test_ce_onehot.horsepower != '?']

train_len = len(df_train_ce_onehot)

df_train_ce_onehot = pd.merge(df_train_ce_onehot,df_test_ce_onehot,how='outer')
df_test_ce_onehot = pd.merge(df_train_ce_onehot,df_test_ce_onehot,how='outer')


train_t = df_train_ce_onehot.loc[:,['mpg']]
train_t = train_t.drop(range(train_len,len(train_t)))
df_train_ce_onehot = df_train_ce_onehot.drop(range(train_len,len(df_train_ce_onehot)))
df_train_ce_onehot = df_train_ce_onehot.drop(['mpg'],axis=1)
df_test_ce_onehot = df_test_ce_onehot.drop(['mpg'],axis=1)
df_train_ce_onehot = df_train_ce_onehot.fillna(int(0))
df_test_ce_onehot = df_test_ce_onehot.fillna(int(0))
print(train_t)
train_x = np.array(df_train_ce_onehot, dtype="float32")
train_t = np.array(train_t,dtype="float32")


#for i in train_t:
    #print(i)

x = Variable(train_x)
t = Variable(train_t)
#"""
model = Model(in_size=len(df_train_ce_onehot.columns))
optimizer = optimizers.Adam()
optimizer.setup(model)

while(1):
    model.cleargrads()
    y = model(x)
    loss = F.mean_squared_error(y,t)
    loss.backward()
    optimizer.update()
    print("loss:",loss.data)
    if(loss.data < 4):
        break

serializers.save_npz("predict_mileage.npz",model)

test_x = np.array(df_test_ce_onehot, dtype="float32")

tx = Variable(test_x)
ty = model(test_x)

#print(ty.data)
ans = [] 
f = open('ans.csv','w')
writer = csv.writer(f, lineterminator='\n')

#ty = np.array(ty,dtype="float32")
for row in ty:
    print(row)
    writer.writerow(row)
f.close()
#"""


