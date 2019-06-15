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
            l1=L.Linear(in_size,54),
            l2=L.Linear(54,27),
            l3=L.Linear(27,16),
            l4=L.Linear(16,8),
            l5=L.Linear(8,4),
            l6=L.Linear(4,1),
        )
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        return self.l6(h)
    

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

    #正規化
def get_data(x,t):
    #教師データ
    train_x, train_t = [], []
    train_path = "train.tsv"
    test_path = "test.tsv"
    csv_file = open(train_path, "r", encoding="utf_8", errors="", newline="\n" )
    f = csv.reader(csv_file, delimiter="\t", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    
    tmp = [[] for i in range (20)]

    zero_count = False
    flag = [False for i in range(20)]
    #教師データ変換

    for row in f:
        train_x.append(row[1:15])
        train_t.append(row[15])

    train_x = np.array(train_x, dtype="float32")
    train_t = np.array(train_t, dtype="float32")

    return train_x,train_t

x,t = [],[]

df_train = pd.read_csv("train.tsv",delimiter='\t')
df_test = pd.read_csv("test.tsv",delimiter='\t')

list_cols = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','Y']
test_list_cols = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']


ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')
test_ce_ohe = ce.OneHotEncoder(cols=test_list_cols,handle_unknown='impute')
df_train_ce_onehot = ce_ohe.fit_transform(df_train)
df_test_ce_onehot = test_ce_ohe.fit_transform(df_test)

train_len = len(df_train_ce_onehot)

del df_train_ce_onehot['id']
del df_test_ce_onehot['id']

df_train_ce_onehot = pd.merge(df_train_ce_onehot,df_test_ce_onehot,how='outer')

train_t = df_train_ce_onehot.loc[:,['Y_1','Y_2']]
train_t = train_t.drop(range(train_len,len(train_t)))
df_train_ce_onehot = df_train_ce_onehot.drop(range(train_len,len(df_train_ce_onehot)))
df_train_ce_onehot = df_train_ce_onehot.drop(["Y_1","Y_2"],axis=1)
df_train_ce_onehot = df_train_ce_onehot.fillna(int(0))

train_x = np.array(df_train_ce_onehot, dtype="float32")
train_t = np.array(train_t,dtype="float32")

tmp = []
for row in train_t:
    if(row[0] > row[1]):
        tmp.append(0)
    else:
        tmp.append(1)
train_t = np.array(tmp,dtype="float32")
train_t = train_t.reshape(train_len,1)

x = Variable(train_x)
t = Variable(train_t)

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
    if(loss.data < 1):
        break

serializers.save_npz("predict_income.npz",model)

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



