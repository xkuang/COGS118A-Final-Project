# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:37:16 2017

@author: Haoshen Hong
"""
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from util import *

dataset = 'car'
train, Y_train, test, Y_test = loadData(dataset)
X_train, X_test = preprocessing(dataset, train, test, 20)
X_CV_train, X_CV_test, Y_CV_train, Y_CV_test = train_test_split(X_train, Y_train, test_size = 0.2)
if dataset == 'adult':
    model = xgb.XGBClassifier(n_estimators=200, 
                              gamma = 0.0, 
                              learning_rate=0.9,
                              max_depth = 3,
                              reg_alpha = 1.3)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()
if dataset == 'mnist':
    model = xgb.XGBClassifier(n_estimators=100,
                              gamma = 0.0,
                              learning_rate = 1.1,
                              max_depth = 6,
                              reg_alpha = 0.0)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()
if dataset == 'car':
    model = xgb.XGBClassifier(n_estimators=1000,
                              gamma = 0.0,
                              learning_rate = 0.1,
                              max_depth = 4,
                              reg_alpha = 0.2)
    model.fit(X_train, Y_train)
    print('Training Acc: ', model.score(X_train, Y_train))
    print('Testing Acc: ', model.score(X_test, Y_test))
    raise SystemExit()

p_range = np.arange(0, 1.5, 0.1)
cv_score = list()
for p in p_range:
    model = xgb.XGBClassifier(n_estimators=1000,
                              gamma = 0.0,
                              learning_rate = 0.1,
                              max_depth = 4,
                              reg_alpha = 0.2)
    score = model.fit(X_CV_train, Y_CV_train).score(X_CV_test, Y_CV_test)
    cv_score.append(score)
    print('For p = ', p, ', CV acc is ', score)
max_p = p_range[np.argmax(cv_score)]

model = xgb.XGBClassifier(n_estimators=1000,
                              gamma = 0.0,
                              learning_rate = 0.1,
                              max_depth = 4,
                              reg_alpha = max_p)
print('Best p is ', max_p)
model.fit(X_train, Y_train)
print('Training Acc: ', model.score(X_train, Y_train))
print('Testing Acc: ', model.score(X_test, Y_test))