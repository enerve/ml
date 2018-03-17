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
    
    X, Y, X_test, Y_test, X_valid, Y_valid = \
        data_util.split_into_train_test_sets(
            X, Y, args.test_portion, args.validation_portion)
    
    print "Training dataset:   %s" % (X.shape, )
    print "Testing dataset(%d): %s" % (args.test_portion, X_test.shape)
    print "Validation dataset(%d): %s" % (args.validation_portion,
                                          X_valid.shape)

    if args.normalize:
        print "Normalizing..."
        util.pre_norm = "n"
        X, X_test, X_valid = data_util.normalize_all(X, X_test, X_valid)

    if True: # Classification
        split_points = [-1, 8, 10]
        n = len(split_points)
        print "Classes (%s) split at:" %(n)
        print split_points[1:]

        Ya = np.zeros(Y.shape, dtype=np.int16)
        for i, spl in enumerate(split_points):
            Ya[Y>spl] = i

        Ya_test = np.zeros(Y_test.shape)
        for i, spl in enumerate(split_points):
            Ya_test[Y_test>spl] = i

        Ya_valid = np.zeros(Y_valid.shape)
        for i, spl in enumerate(split_points):
            Ya_valid[Y_valid>spl] = i

        print "Training set"
        print "  Class 0: ", np.sum(Ya == 0)
        print "  Class 1: ", np.sum(Ya == 1)
        print "  Class 2: ", np.sum(Ya == 2)

        print "Validation set"
        print "  Class 0: ", np.sum(Ya_valid == 0)
        print "  Class 1: ", np.sum(Ya_valid == 1)
        print "  Class 2: ", np.sum(Ya_valid == 2)

        if args.draw_classes_histogram:
            draw_classes_histogram(X, Ya, n)

        if args.bayes:
            print "Bayes classifier..."
            util.pre_alg = "bayes"
            # Gaussian plug-in classifier
            
            gpi_classifier = GaussianPlugInClassifier(X, Ya, n)
                
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(gpi_classifier.classify(X_test, Ya_test)[0])
    
        if args.naive:
            print "Naive bayes classifier..."
            util.pre_alg = "naive"
            # Gaussian naive bayes classifier
            
            naive_classifier = GaussianNaiveClassifier(X, Ya, n)
            
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(naive_classifier.classify(X_test, Ya_test)[0])

        if args.perceptron:
            print "Perceptron..."
            util.pre_alg = "perceptron"
            from ml_lib.perceptron import Perceptron
            
            helper.linear_multiclassify(
                X, Ya, X_test, Ya_test, split_points,
                lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 8000, 0))
#             helper.onevsone_multiclassify(
#                 X, Ya, X_test, Ya_test, len(split_points),
#                 lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 8000, 0))
            
        if args.logistic:
            print "Logistic Regression..."
            util.pre_alg = "logistic"
            from ml_lib.logistic import Logistic
            
#             helper.linear_multiclassify(
#                 X, Ya, X_test, Ya_test, split_points,
#                 lambda X, Y: Logistic(X, Y, step_size=0.001, max_steps=15000,
#                                       reg_constant=0.01))
            helper.onevsone_multiclassify(
                X, Ya, X_test, Ya_test, len(split_points),
                lambda X, Y: Logistic(X, Y, step_size=0.001, max_steps=15000,
                                      reg_constant=0.01))

        if args.knn:
            print "k-Nearest Neighbor..."
            util.pre_alg = "knn"
            from ml_lib.knn import KNN
            
#             knn_classifier = KNN(X, Ya, 10, 3)
#             util.report_accuracy(knn_classifier.classify(X_test, Ya_test))
    
            for k in range(10, 30):
                print "%s-NN" % (k+1)
                knn_classifier = KNN(X, Ya, 1+k, 3)
                util.report_accuracy(knn_classifier.classify(X_test, Ya_test))

        if args.svm:
            print "Support Vector Machine..."
            util.pre_alg = "svm"
            from ml_lib.svm import SVM, RBFKernel
            from ml_lib.svm_sk_svc import SVMSkSVC
            from ml_lib.svm_sk_linear import SVMSkLinear

            run_single_classifier = False
            if run_single_classifier: 
                Yb = np.zeros((Ya.shape[0]))
                Yb[Ya>=i] = 1
                svm_classifier = SVM(X, Yb, 1)#), kernel=RBFKernel(1))
                util.report_accuracy(svm_classifier.classify(X, Yb))
                util.report_accuracy(svm_classifier.classify(X_test, Y_test))
        
                helper.linear_multiclassify(
                    X, Ya, X_valid, Ya_valid, split_points,
                    lambda X, Y, lam=1: SVMSkLinear(X, Y, lam))
             
             
            lam_val = [math.pow(1.5, p+1)*10 for p in range(7)]
            b_val = [(p+1)/40 for p in range(27)]
#             lam_val = [math.pow(1.2, p+1)*10 for p in range(27)]
#             b_val = [(p+2)/20 for p in range(7)]
            
            lmbd_sk = lambda lam, b, X, Y: SVMSkSVC(X, Y, lam, b, kernel='rbf')
            lmbd_my = lambda lam, b, X, Y: SVM(X, Y, lam, kernel=RBFKernel(b))

            pre_svm_alg = "sk" #"ny"
            
            cm, acc_list = helper.onevsall_multiclassify_validation(
                X, Ya, X_valid, Ya_valid, len(split_points),
                lmbd_sk if pre_svm_alg=="sk" else lmbd_my,
                lam_val, b_val)

            
            for i in range(3):#len(acc_list)):
                print "--- Class %d" %(i)
                acc_matrix = np.array(acc_list[i])
#                 acc_matrix = np.genfromtxt(util.prefix() + '995985.0' + "_cv_%d.csv" % (i),
#                                            delimiter=",")
                print acc_matrix
                np.savetxt(util.prefix() + "%s_cv_%d.csv" % (pre_svm_alg, i),
                           acc_matrix, delimiter=",", fmt='%.3f')
                util.plot_accuracies(acc_matrix, b_val, "RBF width b",
                                     lam_val, "%s_class%d_b"%(pre_svm_alg, i))
#                 util.plot_accuracies(acc_matrix.T, lam_val, "Lambda (C)",
#                                      b_val, "class%d_l"%(i))

