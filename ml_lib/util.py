'''
Created on Jan 25, 2018

@author: erw
'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
import heapq as hq


def normalize(X, f_range=None, f_mean=None):
    if f_range is None:
        f_range = np.max(X, axis=0) - np.min(X, axis=0)
    X = X / f_range
#     print (f_range * 100).astype(int)
    if f_mean is None:
        f_mean = np.mean(X)
    X -= f_mean
    return X, f_range, f_mean

def draw_class_histograms(X, Y, num_classes, col):
    for c in range(num_classes):
        i_c = Y == c
        x_c = X[i_c, col]
        plt.hist(x_c, alpha=0.5, label='%s' % (c), bins=50)
    plt.xlabel('Feature %s' % (col))
    plt.ylabel('Frequency')
    plt.show()

# def draw_class_histograms(X_0, X_1, col):
#     x_0 = X_0[:, col]
#     x_1 = X_1[:, col]
#     
#     plt.hist(x_0, alpha=0.5, label='0', bins=50)
#     plt.hist(x_1, alpha=0.5, label='1', bins=50)
#     plt.xlabel('Feature %s' % (col))
#     plt.ylabel('Frequency')
#     plt.show()

def split_into_train_test_sets(X, Y, test_portion):
    # Split into Training and Testing sets    
    train_idx=[]
    test_idx=[]
    print test_portion
    for i in range(X.shape[0]):
        if i % 4 == test_portion:
            test_idx.append(i)
        else:
            train_idx.append(i)
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    X = X[train_idx]
    Y = Y[train_idx]
    return (X, Y, X_test, Y_test)


# Draw class pdf of the classifier but along a single given axis/column
def draw_classes_pdf(X, Y, classifier, threshold, col):
    A, PY = classifier.classify(X, Y, threshold)
    x0 = X[:, col]
    
    plt.plot(x0, PY[1], 'go')
    plt.xlabel('Feature %s' %(col))
    plt.ylabel('Class probability')
    plt.show()

    t = A[0,0] / (A[0, 0] + A[0, 1])    # true negatives
    f = A[1,0] / (A[1, 0] + A[1, 1])    # false negatives
    print('True negatives: %s \t False negatives: %s \t' % (t, f))
    
# draw ROC curve
# true negative vs false negative
def draw_ROC_curve(X_test, Y_test, classifier):
    
    # Heap to keep test_classification_gaussian accuracy measurements sorted by threshold value
    measureHeap = []
    hq.heappush(measureHeap, (0, 1, 1))
    hq.heappush(measureHeap, (1, 0, 0))
    
    # Number of different threshold points to measure at
    measurements = 100
    
    start_time = time.time()

    # Heap to keep track of largest threshold gaps that we can sub-sample within
    heap = [(1, 0, 1, 1, 0)]    # tgap, x1, x2, t1, t2  where tgap = abs(t2-t1)
    # At each iteration, find the largest threshold gap and measure accuracy at 
    # its midpoint threshold value, thereby splitting the gap into two new gaps.
    for i in range(measurements):
    
        tgap, x1, x2, t1, t2 = hq.heappop(heap)
        x = (x1 + x2) / 2   # new threshold
        A, P = classifier.classify(X_test, Y_test, [x, 1-x])
        t = A[0,0] / (A[0, 0] + A[0, 1])    # true negatives
        f = A[1,0] / (A[1, 0] + A[1, 1])    # false negatives
        
        hq.heappush(measureHeap, (x, t, f)) # store measurement for later plotting
    
        hq.heappush(heap, (-abs(t - t1), x1, x, t1, t))
        hq.heappush(heap, (-abs(t2 - t), x, x2, t, t2))
        
    # For plotting, create sorted arrays from measurements heap
    rstep = []
    tn =    []
    fn =    []
    for i in range(measurements):
        x, t, f = hq.heappop(measureHeap)
        rstep.append(x)
        tn.append(t)
        fn.append(f)
    
    elapsed_time = time.time() - start_time
    print 'Avg time: %s' %(elapsed_time / measurements) #0.0409 ... 0.0038
    
    plt.plot(fn, tn, 'r-')
    plt.xlabel('false negatives')
    plt.ylabel('true negatives')
    plt.show()
    
    plt.plot(rstep, tn, 'r-')   
    plt.xlabel('threshold')
    plt.ylabel('true negatives')
    plt.show()
    
    plt.plot(rstep, tn, 'r-')   
    plt.xlabel('threshold')
    plt.ylabel('true negatives')
    plt.xlim(left=0.999)
    plt.show()

    plt.plot(rstep, fn, 'r-')   
    plt.xlabel('threshold')
    plt.ylabel('false negatives')
    plt.show()
    
    # test_classification_gaussian set 0 has fn=0 at > .9998      reasonable at 0.5 (i.e 95% tn)
    # test_classification_gaussian set 1 has fn=0 at > who knows  reasonable at 0.5
    # test_classification_gaussian set 2 has fn=0 at > .999999    reasonable at 0.9995
    # test_classification_gaussian set 3 has fn=0 at > .999999    reasonable at .9995

def draw_classes_data(X, Y, colA, colB):
    xA = X[:, colA]
    xB = X[:, colB]
    
    xA_0 = xA[np.nonzero(1-Y)]
    xB_0 = xB[np.nonzero(1-Y)]
    xA_1 = xA[np.nonzero(Y)]
    xB_1 = xB[np.nonzero(Y)]

    plt.plot(xA_0, xB_0, 'ro')
    plt.plot(xA_1, xB_1, 'go')
    plt.xlabel('Feature %d' % colA)
    plt.ylabel('Feature %d' % colB)
    #plt.ylim(0,0.000001)
    plt.show()
    return plt

def report_accuracy(c_matrix):
    correct = sum([c_matrix[i, i] for i in range(len(c_matrix[0]))])
    
    print "Accuracy: %s%%" % (correct / np.sum(c_matrix))
    for c in c_matrix:
        print "\t", c
    