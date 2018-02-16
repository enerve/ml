'''
@author erwin

'''
from __future__ import division
import numpy as np
import util
from gaussian_plugin_classifier import GaussianPlugInClassifier 
from gaussian_naive_classifier import GaussianNaiveClassifier
import argparse
# from sklearn.linear_model import Perceptron
from perceptron import Perceptron

# Draw histograms for each column
def draw_classes_histogram(X, Y):
    X_0 = X[np.nonzero(1 - Y)] # All y=0 datapoints
    X_1 = X[np.nonzero(Y)] # All y=1 datapoints
    for col in range(10):
        util.draw_class_histograms(X_0, X_1, col)

def features_to_use():
    feature_idx = []
    for k in [0, 1, 2]:
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            feature_idx.append(10*k + 2 + j)
    return feature_idx



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='path to wdbc file')
    parser.add_argument('--test_portion', help='Which portion to use as test set', default=1, type=int)
    parser.add_argument('--draw_classes_data', action='store_true')
    parser.add_argument('--draw_classes_histogram', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--perceptron', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    
    data = np.genfromtxt(args.file, delimiter=",", converters={1: lambda x: 1.0 if x=='M' else 0.0})
    Y = data[:, 1].astype(int)
    X = data[:, features_to_use()]

    X, Y, X_test, Y_test = util.split_into_train_test_sets(X, Y, 1)#args.test_portion)
    print X.shape, X_test.shape

    if args.normalize:
        print "Normalizing..."
        X, f_range, f_mean = util.normalize(X)
        X_test = util.normalize(X_test, f_range, f_mean)[0]
    
    if args.draw_classes_histogram:
        draw_classes_histogram(X, Y)
        
    if args.draw_classes_data:
        util.draw_classes_data(X, Y, 5, 6)

    if args.bayes:
        print "Bayes classifier..."
        # Gaussian plug-in classifier
        gpi_classifier = GaussianPlugInClassifier(X, Y, 2)
        # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(gpi_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])

        # util.draw_ROC_curve(X_test, Y_test, gpi_classifier)
        # util.draw_classes_pdf(X, Y, gpi_classifier, [0.5, 0.5], 3)
    
    if args.naive:
        print "Naive Bayes classifier..."
        # Gaussian naive classifier
        gn_classifier = GaussianNaiveClassifier(X, Y, 2)
        # util.report_accuracy(gn_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(gn_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])
        
        util.draw_ROC_curve(X_test, Y_test, gn_classifier)

    if args.perceptron:
#         perceptron = Perceptron(tol=None, max_iter=5000000)
#         perceptron.fit(X, Y)
#         print perceptron.score(X, Y)
        perceptron = Perceptron(X, Y, args.stochastic, 1, 300000, 0)
        print perceptron.classify(X, Y)
        print perceptron.classify(X_test, Y_test)