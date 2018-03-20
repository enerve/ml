'''
Created on Mar 15, 2018

Utility methods to help munge data and prepare for running ML algorithms

@author: enerve
'''

import logging
import numpy as np

import ml_lib.log as log
import ml_lib.util as util

logger = logging.getLogger(__name__)

def init_logger():
    logger.setLevel(logging.INFO)
    pass

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
    if f_mean is None:
        f_mean = np.mean(X)
    X -= f_mean
    return X, f_range, f_mean

def normalize_all(X, X_valid, X_test):
    X, f_range, f_mean = normalize(X)
    X_test = normalize(X_test, f_range, f_mean)[0]
    X_valid = normalize(X_valid, f_range, f_mean)[0]
    return (X, X_valid, X_test)

def split_into_train_test_sets(X, Y, validation_portion, test_portion):
    # Split into Training and Testing sets
    util.pre_validation_portion = validation_portion
    util.pre_test_portion = test_portion

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
    return (X, Y, X_valid, Y_valid, X_test, Y_test)

def bucketify(Y, Y_valid, Y_test, split_points):
    logger.debug("Classes (%s) split at:", len(split_points))
    logger.debug("%s", split_points[1:])

    Ya = np.zeros(Y.shape, dtype=np.int16)
    for i, spl in enumerate(split_points):
        Ya[Y>spl] = i

    Ya_valid = np.zeros(Y_valid.shape)
    for i, spl in enumerate(split_points):
        Ya_valid[Y_valid>spl] = i

    Ya_test = np.zeros(Y_test.shape)
    for i, spl in enumerate(split_points):
        Ya_test[Y_test>spl] = i
        
    return (Ya, Ya_valid, Ya_test)

def describe_classes(Y, Y_valid, Y_test):
    logger.debug("Training set")
    logger.debug("  Class 0: %d", np.sum(Y == 0))
    logger.debug("  Class 1: %d", np.sum(Y == 1))
    logger.debug("  Class 2: %d", np.sum(Y == 2))

    logger.debug("Validation set")
    logger.debug("  Class 0: %d", np.sum(Y_valid == 0))
    logger.debug("  Class 1: %d", np.sum(Y_valid == 1))
    logger.debug("  Class 2: %d", np.sum(Y_valid == 2))

    logger.debug("Test set")
    logger.debug("  Class 0: %d", np.sum(Y_test == 0))
    logger.debug("  Class 1: %d", np.sum(Y_test == 1))
    logger.debug("  Class 2: %d", np.sum(Y_test == 2))

