'''
Created on Jan 25, 2018

Utility methods to deal with results, display on screen, in figures or 
output to files

@author: enerve
'''
from __future__ import division
import heapq as hq
import logging
import numpy as np
import time

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def init_logger():
    #logger.setLevel(logging.INFO)
    pass

# ------ Drawing ---------

pre_outputdir = None
pre_dataset = None
pre_test_portion = None
pre_validation_portion = None
pre_alg = None
pre_norm = None
pre_tym = None

def prefix_init(args):
    global pre_tym, pre_outputdir
    pre_tym = str(int(round(time.time()) % 1000000))
    pre_outputdir = args.output_dir

def prefix(other_tym=None):
    return (pre_outputdir if pre_outputdir else '') + \
        ("%s_"%pre_dataset if pre_dataset else '') + \
        ("v%s_"%pre_validation_portion if pre_validation_portion else '') + \
        ("t%s_"%pre_test_portion if pre_test_portion else '') + \
        ("%s_"%pre_alg if pre_alg else '') + \
        ("%s_"%(other_tym or pre_tym)) + \
        ("%s_"%pre_norm if pre_norm else '')


def draw_class_histograms(X, Y, num_classes, col):
    for c in range(num_classes):
        i_c = Y == c
        x_c = X[i_c, col]
        plt.hist(x_c, alpha=0.5, label='%s' % (c), bins=50)
    plt.xlabel('Feature %s' % (col))
    plt.ylabel('Frequency')
    plt.show()

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
    logger.debug('True negatives: %s \t False negatives: %s \t', t, f)
    
# draw ROC curve
# true negative vs false negative
def draw_ROC_curve(X_test, Y_test, classifier):
    
    # Heap to keep test_classification_gaussian accuracy measurements 
    # sorted by threshold value
    measureHeap = []
    hq.heappush(measureHeap, (0, 1, 1))
    hq.heappush(measureHeap, (1, 0, 0))
    
    # Number of different threshold points to measure at
    measurements = 100
    
    start_time = time.time()

    # Heap to keep track of largest threshold gaps that we can subsample within
    heap = [(1, 0, 1, 1, 0)]    # tgap, x1, x2, t1, t2  where tgap = abs(t2-t1)
    # At each iteration, find the largest threshold gap and measure accuracy at 
    # its midpoint threshold value, thereby splitting the gap into 2 new gaps.
    for i in range(measurements):
    
        tgap, x1, x2, t1, t2 = hq.heappop(heap)
        x = (x1 + x2) / 2   # new threshold
        A, P = classifier.classify(X_test, Y_test, [1-x, x])
        t = A[0,0] / (A[0, 0] + A[0, 1])    # true negatives
        f = A[1,0] / (A[1, 0] + A[1, 1])    # false negatives
        
        hq.heappush(measureHeap, (x, t, f)) # store for later plotting
    
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
    logger.debug('Avg time: %s', elapsed_time / measurements)
    
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

def save_plot(pref=None):
    fname = prefix() \
        + ("%s_"%pref if pref else '') \
        + 'val.png'
    plt.savefig(fname, bbox_inches='tight')
    logger.debug(fname)

def hist(x, x_label, bins=100, pref=None):
    logger.debug("%s: %s", x_label, x.shape[0])
    plt.hist(x, bins)
    plt.xlabel(x_label)
    save_plot(pref)
    #plt.show()

def plot(x, y, x_label, y_label, pref=None):
    plt.plot(x, y, 'b-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    save_plot(pref)
    #plt.show()
    plt.clf() # clear figure

def plot_all(x, ys, x_label, ys_label, labels=None, pref=None):
    for i, y in enumerate(ys):
        plt.plot(x, y, '-', label=labels[i] if labels is not None else '')
    plt.xlabel(x_label)
    plt.ylabel(ys_label)
    if labels is not None:
        plt.legend(loc=4)
    save_plot(pref)
    plt.show()
    plt.clf() # clear figure

def plot_accuracy(acc, x_values, line_labels=None, pref=None):
    plot(x_values, acc, 'C or lambda', 'Validation accuracy', pref)

def plot_accuracies(acc_matrix, x_values, x_label, z_labels=None,
                    z_title = None, pref=None):
    for i, acc in enumerate(acc_matrix):
        plt.plot(x_values, acc, '-', label="%0.2f" %(z_labels[i]))

    plt.xlabel(x_label)
    plt.ylabel('Validation accuracy')
    plt.legend(loc=4, title=z_title)
    save_plot(pref)
#     plt.show()
    plt.clf() # clear figure

def plot_validation_results(val_acc, xlabel, ylabel, pref=None):
    x_scatter = [x[0] for x in val_acc]
    y_scatter = [x[1] for x in val_acc]
    marker_size = 100
    
    # plot validation accuracy
    colors = [val_acc[x] for x in val_acc]
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('validation accuracy')
    save_plot(pref)
    plt.show()

# ------ Logging/Debugging ---------

def report_accuracy(c_matrix, display_matrix=True):
    logger.info("Accuracy: %0.2f%%", get_accuracy(c_matrix))
    if display_matrix:
        for c in c_matrix:
            logger.info("\t%s", c)
    
def get_accuracy(c_matrix):
    correct = sum([c_matrix[i, i] for i in range(len(c_matrix[0]))])
    
    return 100 * (correct / np.sum(c_matrix))
    
def confusion_matrix(Y_actual, Y_guess, n):
    c_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c_matrix[i, j] = np.sum(
                np.logical_and((Y_actual == i), (Y_guess == j)))
    return c_matrix