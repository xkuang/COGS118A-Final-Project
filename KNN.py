# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:37:16 2017

@author: Haoshen Hong
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from util import *

dataset = 'car'
train, Y_train, test, Y_test = loadData(dataset)
X_train, X_test = preprocessing(dataset, train, test, 20)
X_CV_train, X_CV_test, Y_CV_train, Y_CV_test = train_test_split(X_train, Y_train, test_size = 0.2)

if dataset == 'adult':
    model = KNeighborsClassifier(21)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()
if dataset == 'mnist':
    model = KNeighborsClassifier(3)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()
if dataset == 'car':
    model = KNeighborsClassifier(9)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()
k_range = np.arange(1, 31, 2)
cv_score = list()
for k in k_range:
    model = KNeighborsClassifier(k)
    score = model.fit(X_CV_train, Y_CV_train).score(X_CV_test, Y_CV_test)
    cv_score.append(score)
    print('For k = ', k, ', CV acc is ', score)
max_k = k_range[np.argmax(cv_score)]
model = KNeighborsClassifier(max_k)
model.fit(X_train, Y_train)
print('Best k is ', max_k)
print('Training Acc: ', model.score(X_train, Y_train))
print('Testing Acc: ', model.score(X_test, Y_test))