'''
@author enerve

'''
from __future__ import division
import logging
import math
import numpy as np

import ml_lib.util as util
import ml_lib.data_util as data_util
import ml_lib.helper as helper
import ml_lib.cmd_line as cmd_line
import ml_lib.log as log

# Draw histograms for each column
def draw_classes_histogram(X, Y):
    X_0 = X[np.nonzero(1 - Y)] # All y=0 datapoints
    X_1 = X[np.nonzero(Y)] # All y=1 datapoints
    for col in range(10):
        util.draw_class_histograms(X_0, X_1, 2, col)

def features_to_use():
    feature_idx = []
    for k in [0, 1, 2]:
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            feature_idx.append(10*k + 2 + j)
    return feature_idx

def main():
    args = cmd_line.parse_args()
    
    util.prefix_init(args)
    util.pre_dataset = "wdbc"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log.configure_logger(logger, util.pre_dataset)

    logger.info("--- WDBC dataset ---")

    data = np.genfromtxt(args.file, delimiter=",",
                         converters={1: lambda x: 1.0 if x=='M' else 0.0})
    Y = data[:, 1].astype(int)
    X = data[:, features_to_use()]

    X, Y, X_valid, Y_valid, X_test, Y_test = \
        data_util.split_into_train_test_sets(X, Y, None, args.test_portion)
    
    logger.debug("%s %s", X.shape, X_test.shape)

    if args.normalize:
        logger.info("Normalizing...")
        util.pre_norm = "n"
        X, X_valid, X_test = data_util.normalize_all(X, X_valid, X_test)
    
    if args.draw_classes_histogram:
        draw_classes_histogram(X, Y)
        
    if args.draw_classes_data:
        util.draw_classes_data(X, Y, 5, 6)

    if args.bayes:
        logger.info("Bayes classifier...")
        util.pre_alg = "bayes"
        from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
        # Gaussian plug-in classifier
        gpi_classifier = GaussianPlugInClassifier(X, Y, 2)
        # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(
            gpi_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])

        util.draw_ROC_curve(X_test, Y_test, gpi_classifier)
        # util.draw_classes_pdf(X, Y, gpi_classifier, [0.5, 0.5], 3)
    
    if args.naive:
        logger.info("Naive Bayes classifier...")
        util.pre_alg = "naive"
        from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier
        # Gaussian naive classifier
        gn_classifier = GaussianNaiveClassifier(X, Y, 2)
        # util.report_accuracy(gn_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(
            gn_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])
        
        util.draw_ROC_curve(X_test, Y_test, gn_classifier)

    if args.sklearn_perceptron:
        logger.info("Scikit-learn Perceptron...")
        util.pre_alg = "scikitperceptron"
        from sklearn.linear_model import Perceptron
        perceptron = Perceptron(tol=None, max_iter=300000)
        perceptron.fit(X, Y)
        logger.info("Mean accuracy: %s%%", 100 * perceptron.score(X, Y))

    if args.perceptron:
        logger.info("Perceptron...")
        util.pre_alg = "perceptron"
        from ml_lib.perceptron import Perceptron

        Ya = Y
        Ya_test = Y_test
        split_points = [-1, 0]
        n = len(split_points)
        
        helper.onevsone_multiclassify(
            X, Ya, X_test, Ya_test, n,
            lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 30000, 0))
            
    if args.logistic:
        logger.info("Logistic Regression...")
        util.pre_alg = "logistic"
        from ml_lib.logistic import Logistic
        
        
        Ya = Y
        Ya_test = Y_test
        split_points = [-1, 0]
        n = len(split_points)

        helper.onevsone_multiclassify(
            X, Ya, X_test, Ya_test, n,
            lambda X, Y: Logistic(X, Y, step_size=0.001, max_steps=15000,
                                  reg_constant=1))
        

    if args.knn:
        logger.info("k-Nearest Neighbor...")
        util.pre_alg = "knn"
        from ml_lib.knn import KNN
        
        k_range = 10
        p_range = 6 # / 2.0
        a_matrix = np.zeros((k_range, p_range))
        for k in range(k_range):
            logger.info("%s-NN", k+1)
            for p in range(p_range):
                knn_classifier = KNN(X, Y, 1+k, dist_p=(p+1)/2.0)
                a_matrix[k, p] = util.get_accuracy(
                    knn_classifier.classify(X_test, Y_test))

        logger.info("%s", a_matrix)

    if args.svm:
        logger.info("Support Vector Machine...")
        util.pre_alg = "svm"
        from ml_lib.svm import SVM, RBFKernel

        #lam_val = [math.pow(1.2, p) for p in range(-10,20)]
        lam_val = [p/10 for p in range(1,350)]

        acc = np.zeros(len(lam_val))
        for i, lam in enumerate(lam_val):
            svm_classifier = SVM(X, Y, lam)#), kernel=RBFKernel(1))
            #util.report_accuracy(svm_classifier.classify(X, Y))
            cm = svm_classifier.classify(X_test, Y_test)
            util.report_accuracy(cm)
            acc[i] = util.get_accuracy(cm)

        logger.info("\nAccuracies found for lambda:")
        for i, lam in enumerate(lam_val):
            logger.info("%f: \t%f", lam, acc[i])
        util.plot_accuracy(acc, lam_val)

if __name__ == '__main__':
    main()