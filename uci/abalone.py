'''
@author erw

'''
from __future__ import division

import argparse
import numpy as np

from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier

import ml_lib.util as util
import math 
import time


# Draw histograms for each column
def draw_classes_histogram(X, Y, num_classes):
    for col in range(10):
        util.draw_class_histograms(X, Y, num_classes, col)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()  
    parser.add_argument('file', help='path to data file')
    parser.add_argument('--output_dir', help='path to store output files')
    parser.add_argument('--test_portion',
                        help='Which portion to use as test set',
                        default=1, type=int)
    parser.add_argument('--validation_portion',
                        help='Which portion to use as validation set',
                        default=2, type=int)
    parser.add_argument('--draw_classes_data', action='store_true')
    parser.add_argument('--draw_classes_histogram', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--perceptron', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--logistic', action='store_true')
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--svm', action='store_true')
    args = parser.parse_args()

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
#     X[:, 2][data[:, 0] == 2.0] = 1
    
    X = util.append_feature(data[:, 1] * data[:, 2] * data[:, 3], X)
    X = util.append_feature(data[:, 1] * data[:, 2], X)
    X = util.append_feature(data[:, 1] * data[:, 1], X)
        
    # X = util.select_features(X, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    Y = data[:, 8].astype(int)

    util.pre_outputdir = args.output_dir

    print "Full dataset: ", X.shape
    print "  Males:   ", np.sum(X[:,0])
    print "  Females: ", np.sum(X[:,1])
    print "  Infants: ", np.sum(X[:,2])
    
    X, Y, X_test, Y_test, X_valid, Y_valid = util.split_into_train_test_sets(
        X, Y, args.test_portion, args.validation_portion)
    
    print "Training dataset:   %s" % (X.shape, )
    print "Testing dataset(%d): %s" % (args.test_portion, X_test.shape)
    print "Validation dataset(%d): %s" % (args.validation_portion,
                                          X_valid.shape)

    if args.normalize:
        print "Normalizing..."
        util.pre_norm = "n"
        X, f_range, f_mean = util.normalize(X)
        X_test = util.normalize(X_test, f_range, f_mean)[0]
        X_valid = util.normalize(X_valid, f_range, f_mean)[0]

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
            
            def create_classifier(X, Y):
                return Perceptron(X, Y, args.stochastic,
                                        1, 8000, 0)
            
            util.linear_multiclassify(X, Ya, X_test, Ya_test,
                                      split_points, create_classifier)
            
        if args.logistic:
            print "Logistic Regression..."
            util.pre_alg = "logistic"
            from ml_lib.logistic import Logistic
            
            def create_classifier(X, Y):
                return Logistic(X, Y, step_size=0.001, max_steps=15000,
                                    reg_constant=0.01)
                
            util.linear_multiclassify(X, Ya, X_test, Ya_test,
                                      split_points, create_classifier)

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

#             Yb = np.zeros((Ya.shape[0]))
#             Yb[Ya>=i] = 1
#             svm_classifier = SVM(X, Yb, 1)#), kernel=RBFKernel(1))
#             util.report_accuracy(svm_classifier.classify(X, Yb))
#             util.report_accuracy(svm_classifier.classify(X_test, Y_test))
  
#             util.linear_multiclassify(X, Ya, X_valid, Ya_valid, split_points, 
#                                       lambda X, Y, lam=1: SVMSkLinear(X, Y, lam))
            
#             cm = util.onevsall_multiclassify(
# #                         X, Ya, X, Ya, len(split_points),
#                 X, Ya, X_valid, Ya_valid, len(split_points),
#                 lambda X, Y, lam=1000, b=0.1: SVMSkSVC(X, Y, lam, b,
#                                                     kernel='rbf'))
#             util.report_accuracy(cm)
            
            lam_val = [math.pow(2, p-3) for p in range(8)]                 
#             lam_val = [(p+1)/10 for p in range(8)]
            b_val = [(p+1)/10 for p in range(10)]

            cm = util.onevsall_multiclassify_validation(
                X, Ya, X_valid, Ya_valid, len(split_points),
                lambda lam, b, X, Y: SVMSkSVC(X, Y, lam, b, kernel='rbf'),
                lam_val, b_val)
     
#             acc = np.zeros((len(lam_val), len(b_val)))
#             acc2 = np.zeros((len(lam_val), len(b_val)))
#             for i, lam in enumerate(lam_val):
#                 print "============== lam = %f ===========" %(lam)
#                 for j, b in enumerate(b_val):
#                     print "-------------- b = %f ----------" %(b)
# #                     cm = util.onevsall_multiclassify(
# #                         X, Ya, X_valid, Ya_valid, len(split_points),
# #                         lambda X, Y, lam=lam, b=b: SVM(X, Y, lam,
# #                                                        kernel=RBFKernel(b)))
# #                     acc[i,j] = util.get_accuracy(cm)
#                     cm = util.onevsall_multiclassify(
# #                         X, Ya, X, Ya, len(split_points),
#                         X, Ya, X_valid, Ya_valid, len(split_points),
#                         lambda X, Y, lam=lam, b=b: SVMSkSVC(X, Y, lam, b,
#                                                             kernel='rbf'))
#  
# #                     Yb = np.zeros((Ya.shape[0]))
# #                     Yb[Ya==i] = 1
# #                     classifier = create_classifier(X, Yb)
# #                     print classifier.classify(X, Yb)
# #                     Yp_i = classifier.predict(X_test)
#                      
#                     acc2[i,j] = util.get_accuracy(cm)
#      
#             print "\nAccuracies found for lambda:"
#             print lam_val
#             print b_val
#             print acc
#             print acc2
#             tym = str(time.time() % 1000000)
#             np.savetxt(util.prefix() + tym + "_cv.csv", acc,
#                        delimiter=",", fmt='%.3f')
#             np.savetxt(util.prefix() + tym + "_2_cv.csv", acc2,
#                        delimiter=",", fmt='%.3f')
# #             for i, lam in enumerate(lam_val):
# #                 for j, b in enumerate(b_val):
# #                     print "%f, %d: \t%f" %(lam, b, acc[i])
# #             util.plot_accuracy(acc, lam_val)
