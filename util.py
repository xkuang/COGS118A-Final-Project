# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:53:41 2017

@author: Haoshen Hong
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from mnist import read, show
from sklearn.cross_validation import train_test_split

def loadData(data):
    if data == 'adult':
        train = pd.read_csv('./adult/adult.data', header = None)
        test = pd.read_csv('./adult/adult.test', header = None)
        
        train[14] = train[14].str.replace('<=50K', '0')
        train[14] = train[14].str.replace('>50K', '1')
        Y_train = train[14] = train[14].astype('int')
        del train[14]
                
        test[14] = test[14].str.replace('<=50K.', '0')
        test[14] = test[14].str.replace('>50K.', '1')
        Y_test = test[14] = test[14].astype('int')
        del test[14]
        
        return train, Y_train, test, Y_test
    
    if data == 'mnist':
        train = list(read("training", path="./mnist/"))
        X_train = [pair[1] for pair in train]
        X_train = np.array(X_train).reshape((-1, 28*28))
        Y_train = [pair[0] for pair in train]
        Y_train = np.array(Y_train).reshape((-1,))
        
        test = list(read("testing", path="./mnist/"))
        X_test = [pair[1] for pair in test]
        X_test = np.array(X_test).reshape((-1, 28*28))
        Y_test = [pair[0] for pair in test]
        Y_test = np.array(Y_test).reshape((-1,))
        return X_train, Y_train, X_test, Y_test
        
    if data == 'car':
        train = pd.read_csv('./car/car.data', header = None)
        label = train[6]
        del train[6]
        
        X_train, X_test, Y_train, Y_test = train_test_split(
                train, label, test_size = 0.2)

        Y_train = Y_train.replace(['unacc', 'acc', 'good', 'vgood'],
                        [1, 2, 3, 4])
        Y_test = Y_test.replace(['unacc', 'acc', 'good', 'vgood'],
                        [1, 2, 3, 4])
        return X_train, Y_train, X_test, Y_test
    
def preprocessing(data, train, test, n_comp):
    if data == 'adult':
        X_train, X_test = onehot(train, test)
        
        p = PCA(n_comp)
        p.fit(X_train)
        X_train, X_test = p.transform(X_train), p.transform(X_test)
        
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        return X_train, X_test
    
    if data == 'mnist':
        p = PCA(n_comp)
        p.fit(train)
        X_train, X_test = p.transform(train), p.transform(test)
        
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        return X_train, X_test
    
    if data == 'car':
        train.loc[:,0] = train.loc[:,0].replace(['vhigh', 'high', 'med', 'low'],
                        [1, 2, 3, 4])
        test.loc[:,0] = test.loc[:,0].replace(['vhigh', 'high', 'med', 'low'],
                        [1, 2, 3, 4])
        train.loc[:,1] = train.loc[:,1].replace(['vhigh', 'high', 'med', 'low'],
                        [1, 2, 3, 4])
        test.loc[:,1] = test.loc[:,1].replace(['vhigh', 'high', 'med', 'low'],
                        [1, 2, 3, 4])
        train.loc[:,3] = train.loc[:,3].replace(['2', '4', 'more'],
                        [1, 2, 3])
        test.loc[:,3] = test.loc[:,3].replace(['2', '4', 'more'],
                        [1, 2, 3])
        train.loc[:,4] = train.loc[:,4].replace(['small', 'med', 'big'],
                        [1, 2, 3])
        test.loc[:,4] = test.loc[:,4].replace(['small', 'med', 'big'],
                        [1, 2, 3])
        train.loc[:,5] = train.loc[:,5].replace(['low', 'med', 'high'],
                        [1, 2, 3])
        test.loc[:,5] = test.loc[:,5].replace(['low', 'med', 'high'],
                        [1, 2, 3])
        X_train, X_test = onehot(train, test)
        return X_train, X_test
        
        
def onehot(train, test):
    for cols in train.columns:
        if train[cols].dtype == np.object:
            train = pd.concat((train, pd.get_dummies(train[cols], prefix=cols)), axis=1)
            del train[cols]

    for cols in test.columns:
        if test[cols].dtype == np.object:
            test = pd.concat((test, pd.get_dummies(test[cols], prefix=cols)), axis=1)
            del test[cols]
                
    col = train.columns
    col_test = test.columns
    for index in col:
        if index in col_test:
            pass
        else:
            del train[index]
    
    col = train.columns
    col_test = test.columns
    for index in col_test:
        if index in col:
            pass
        else:
            del test[index]
    return train, test