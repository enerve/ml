'''
@author erw

'''
from __future__ import division

import argparse
import numpy as np

from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier

import ml_lib.util as util
import ml_lib.data_util as data_util
import ml_lib.helper as helper
import ml_lib.cmd_line as cmd_line

import math


# Draw histograms for each column
def draw_classes_histogram(X, Y, num_classes):
    for col in range(10):
        util.draw_class_histograms(X, Y, num_classes, col)


if __name__ == '__main__':
    args = cmd_line.parse_args()

    util.prefix_init()

    print "--- Abalone dataset ---"
    util.pre_dataset = "abalone"

    data = np.genfromtxt(args.file, delimiter=",",
                         converters={0: lambda x: 0.0 if x=='M' 
                                     else 1.0 if x=='F' 
                                     else 2.0 if x=='I' else -1.0}
                        )
    
    # Create separate column for each categorical 'sex' value
    X = np.append(np.zeros((data.shape[0], 2)), data[:,1:8], axis=1)    
    X[:, 0][data[:, 0] == 0.0] = 1
    X[:, 1][data[:, 0] == 1.0] = 1
    
    X = data_util.append_feature(data[:, 1] * data[:, 2] * data[:, 3], X)
    X = data_util.append_feature(data[:, 1] * data[:, 2], X)
    X = data_util.append_feature(data[:, 1] * data[:, 1], X)

    # X = util.select_features(X, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    Y = data[:, 8].astype(int)

    util.pre_outputdir = args.output_dir

    print "Full dataset: ", X.shape
    print "  Males:   ", np.sum(X[:,0])
    print "  Females: ", np.sum(X[:,1])
    print "  Infants: ", np.sum(X[:,2])
    
    X, Y, X_valid, Y_valid, X_test, Y_test = \
        data_util.split_into_train_test_sets(
            X, Y, args.validation_portion, args.test_portion)
    
    print "Training dataset:   %s" % (X.shape, )
    print "Validation dataset(%d): %s" % (args.validation_portion,
                                          X_valid.shape)
    print "Testing dataset(%d): %s" % (args.test_portion, X_test.shape)

    if args.normalize:
        print "Normalizing..."
        util.pre_norm = "n"
        X, X_valid, X_test = data_util.normalize_all(X, X_valid, X_test)

    if True: # Classification
        split_points = [-1, 8, 10]

        Ya, Ya_valid, Ya_test = data_util.bucketify(Y, Y_valid, Y_test,
                                                    split_points)
        data_util.describe_classes(Ya, Ya_valid, Ya_test)
        num_classes = len(split_points)

        if args.draw_classes_histogram:
            draw_classes_histogram(X, Ya, num_classes)

        if args.bayes:
            print "Bayes classifier..."
            util.pre_alg = "bayes"
            # Gaussian plug-in classifier
            
            gpi_classifier = GaussianPlugInClassifier(X, Ya, num_classes)
                
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(gpi_classifier.classify(X_test, Ya_test)[0])
    
        if args.naive:
            print "Naive bayes classifier..."
            util.pre_alg = "naive"
            # Gaussian naive bayes classifier
            
            naive_classifier = GaussianNaiveClassifier(X, Ya, num_classes)
            
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(naive_classifier.classify(X_test, Ya_test)[0])

        if args.perceptron:
            print "Perceptron..."
            util.pre_alg = "perceptron"
            from ml_lib.perceptron import Perceptron
            
            helper.onevsone_multiclassify(
                X, Ya, X_test, Ya_test, num_classes,
                lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 8000, 0))
            
        if args.logistic:
            print "Logistic Regression..."
            util.pre_alg = "logistic"
            from ml_lib.logistic import Logistic
            
            helper.onevsone_multiclassify(
                X, Ya, X_test, Ya_test, num_classes,
                lambda X, Y: Logistic(X, Y, step_size=0.001, max_steps=15000,
                                      reg_constant=0.01))

        if args.knn:
            print "k-Nearest Neighbor..."
            util.pre_alg = "knn"
            from ml_lib.knn import KNN
            
            run_single_classifier = False
            if run_single_classifier: 
                knn_classifier = KNN(X, Ya, 10, 3)
                util.report_accuracy(knn_classifier.classify(X_test, Ya_test))
    
            for k in range(10, 30):
                print "%s-NN" % (k+1)
                knn_classifier = KNN(X, Ya, 1+k, 3)
                util.report_accuracy(knn_classifier.classify(X_valid, Ya_valid))

        if args.svm:
            print "Support Vector Machine..."
            util.pre_alg = "svm"
            from ml_lib.svm import SVM, RBFKernel
            from ml_lib.svm_sk_svc import SVMSkSVC
            from ml_lib.svm_sk_linear import SVMSkLinear

            run_single_classifier = False
            if run_single_classifier: 
                i = 2
                Yb = np.zeros((Ya.shape[0]))
                Yb[Ya==i] = 1
                svm_classifier = SVM(X, Yb, 100, kernel=RBFKernel(0.2))
                Yb_test = np.zeros((Ya_test.shape[0]))
                Yb_test[Ya_test==i] = 1
                util.report_accuracy(svm_classifier.classify(X_test, Yb_test))

            run_single_multiclassifier = False
            if run_single_multiclassifier:
                helper.onevsone_multiclassify(
                    X, Ya, X_test, Ya_test, num_classes,
                    #lambda X, Y, lam=1: SVMSkLinear(X, Y, lam))
                    lambda X, Y, lam=1: SVM(X, Y, lam))
                    #lambda X, Y, lam=1, b=0.5: SVMSkSVC(X, Y, lam, b, kernel='rbf'))
             
            run_cross_validation = True
            if run_cross_validation:
                for reps in range(4):
                    if reps < 2:
                        lam_val = [math.pow(1.5, p+1)*10 for p in range(7)]
                        b_val = [(p+1)/40 for p in range(27)]
                    else:
                        lam_val = [math.pow(1.2, p+1)*10 for p in range(27)]
                        b_val = [(p+2)/20 for p in range(7)]
                    
                    lmbd_sk = lambda lam, b, X, Y: SVMSkSVC(X, Y, lam, b, kernel='rbf')
                    lmbd_my = lambda lam, b, X, Y: SVM(X, Y, lam, kernel=RBFKernel(b))
        
                    if reps % 2 == 0:
                        pre_svm_alg = "sk"
                    else:
                        pre_svm_alg = "my"
                    
                    cm, acc_list = helper.onevsall_multiclassify_validation(
                        X, Ya, X_valid, Ya_valid, num_classes,
                        lmbd_sk if pre_svm_alg=="sk" else lmbd_my,
                        lam_val, b_val)
        
                    for i in range(num_classes):
                        print "--- Class %d" %(i)
                        acc_matrix = np.array(acc_list[i])
                        #acc_matrix = np.genfromtxt(util.prefix() +
                        #                           '995985.0' + "_cv_%d.csv" % (i),
                        #                           delimiter=",")
                        print acc_matrix
                        np.savetxt(util.prefix() + "%s_cv_%d.csv" % (pre_svm_alg, i),
                                   acc_matrix, delimiter=",", fmt='%.3f')
                        if reps < 2:
                            util.plot_accuracies(acc_matrix, b_val, "RBF width b",
                                                lam_val, "%s_class%d_b"%(pre_svm_alg, i))
                        else:
                            util.plot_accuracies(acc_matrix.T, lam_val, "Lambda (C)",
                                                b_val, "class%d_l"%(i))
    

