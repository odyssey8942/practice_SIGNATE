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
from chainer.datasets import LabeledImageDataset
#from chainercv.transforms import resize
from chainer.datasets import TransformDataset
import chainer.functions as F
import chainer.links as L
import csv
import category_encoders as ce 

#train,test = chainer.datasets.get_mnist(ndim=3)
dataset = LabeledImageDataset('train_master.txt', 'train_images')

def transform(in_data):
    img, label = in_data
    #img = resize(img, (96, 96))
    return img, label

dataset = TransformDataset(dataset, transform)

#print(dataset[1])
split_at = int(len(dataset) * 0.8)
train, test = chainer.datasets.split_dataset(dataset, split_at)

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1=L.Convolution2D(3, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            fc1=L.Linear(6050, 605),
            fc2=L.Linear(605, 10),
            )
    def __call__(self, x, train=True):
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu,2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 4)
        h = F.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)
#"""
model = L.Classifier(Model())
optimizer = optimizers.Adam()
optimizer.setup(model)

batchsize = 1000

def conv(batch,batchsize):
    x,t=[],[]
    for j in range(batchsize):
        x.append(batch[j][0])
        t.append(batch[j][1])
    return Variable(np.array(x)),Variable(np.array(t))

for n in range(20):
    for i in chainer.iterators.SerialIterator(train,batchsize,repeat=False):
        x,t = conv(i,batchsize)

        model.cleargrads()
        loss = model(x,t)
        loss.backward()
        print(loss.data)
        optimizer.update()

    i = chainer.iterators.SerialIterator(test, batchsize).next()
    x,t = conv(i,batchsize)
    loss = model(x,t)
    print(n,loss.data)
    #"""

serializers.save_npz("labeling_picture_10.npz",model)