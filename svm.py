# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:56:46 2017

@author: Haoshen Hong
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from util import *

dataset = 'car'
train, Y_train, test, Y_test = loadData(dataset)
X_train, X_test = preprocessing(dataset, train, test, 20)

if dataset == 'adult':
    model = SVC(256, 'linear')
    test_score = model.fit(X_train, Y_train).score(X_test, Y_test)
    train_score = model.score(X_train, Y_train)
    print("Testing acc ", test_score)
    print("Training acc ", train_score)
    raise SystemExit()
if dataset == 'mnist':
    n_estimators = 10
    model = OneVsRestClassifier(BaggingClassifier(SVC(256, 'linear'),
                    max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    test_score = model.fit(X_train, Y_train).score(X_test, Y_test)
    train_score = model.score(X_train, Y_train)
    print("Testing acc ", test_score)
    print("Training acc ", train_score)
    raise SystemExit()
if dataset == 'car':
    model = OneVsRestClassifier(SVC(8, 'linear'))
    test_score = model.fit(X_train, Y_train).score(X_test, Y_test)
    train_score = model.score(X_train, Y_train)
    print("Testing acc ", test_score)
    print("Training acc ", train_score)
    raise SystemExit()
    
power = 10
C_range = np.power(2.0 , range(-power, power))
scores = list()
X_CV_train, X_CV_test, Y_CV_train, Y_CV_test = train_test_split(X_train, 
                                                Y_train, test_size = 0.2)
for C in C_range:
    model = OneVsRestClassifier(SVC(C, 'linear'))
    score = model.fit(X_CV_train, Y_CV_train) \
                      .score(X_CV_test, Y_CV_test)
    scores.append(score)
    print("C value is ", C)
    print("with score ", score)

max_C = C_range[np.argmax(scores)]
print("Optimal C value is ", max_C)
print("C CV accuracy is ", np.max(scores))

model = OneVsRestClassifier(SVC(max_C, 'linear'))
test_score = model.fit(X_train, Y_train).score(X_test, Y_test)
train_score = model.score(X_train, Y_train)
print("Testing acc ", test_score)
print("Training acc ", train_score)