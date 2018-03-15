'''
Created on Mar 15, 2018

Utility methods to help munge data and prepare for running ML algorithms

@author: erw
'''

import numpy as np

def select_features(X, feature_ids):
    X_selected = np.zeros((X.shape[0], 1))
    for col in feature_ids:
        X_selected = np.append(X_selected, X[:, col:(col+1)], axis=1)
    return X_selected[:, 1:]

def append_feature(V, X):
    return np.append(X, np.reshape(V, (V.shape[0], 1)), axis=1)

def normalize(X, f_range=None, f_mean=None):
    if f_range is None:
        f_range = np.max(X, axis=0) - np.min(X, axis=0)
    X = X / f_range
#     print (f_range * 100).astype(int)
    if f_mean is None:
        f_mean = np.mean(X)
    X -= f_mean
    return X, f_range, f_mean

def split_into_train_test_sets(X, Y, test_portion, validation_portion):
    # Split into Training and Testing sets
    global pre_portion
    pre_portion = test_portion
    train_idx=[]
    test_idx=[]
    valid_idx=[]
    for i in range(X.shape[0]):
        if i % 4 == test_portion:
            test_idx.append(i)
        elif (validation_portion is not None) and i % 4 == validation_portion:
            valid_idx.append(i)
        else:
            train_idx.append(i)
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    X_valid = X[valid_idx]
    Y_valid = Y[valid_idx]
    X = X[train_idx]
    Y = Y[train_idx]
    return (X, Y, X_test, Y_test, X_valid, Y_valid)
