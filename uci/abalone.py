'''
@author erw

'''

import argparse
import numpy as np

from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier

import ml_lib.util as util

# Draw histograms for each column
def draw_classes_histogram(X, Y, num_classes):
    for col in range(10):
        util.draw_class_histograms(X, Y, num_classes, col)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='path to data file')
    parser.add_argument('--test_portion',
                        help='Which portion to use as test set',
                        default=1, type=int)
    parser.add_argument('--draw_classes_data', action='store_true')
    parser.add_argument('--draw_classes_histogram', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--perceptron', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

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
    
    Y = data[:, 8]
    X = util.select_features(X, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    print "Full dataset: ", X.shape
    print "  Males:   ", np.sum(X[:,0])
    print "  Females: ", np.sum(X[:,1])
    print "  Infants: ", np.sum(X[:,2])
    
    X, Y, X_test, Y_test = util.split_into_train_test_sets(
        X, Y, args.test_portion)
    
    print "Training dataset:   %s" % (X.shape, )
    print "Testing dataset(%d): %s" % (args.test_portion, X_test.shape)

    if True:#args.normalize:
        print "Normalizing..."
        X, f_range, f_mean = util.normalize(X)
        X_test = util.normalize(X_test, f_range, f_mean)[0]


    if True: # Classification
        split_points = [-1, 8, 10]
        print "Classes (%s) split at:" %(len(split_points))
        print split_points[1:]

        Ya = np.zeros(Y.shape)
        for i, spl in enumerate(split_points):
            Ya[Y>spl] = i

        Ya_test = np.zeros(Y_test.shape)
        for i, spl in enumerate(split_points):
            Ya_test[Y_test>spl] = i

#         print Ya==2
#         print "Training set"
#         print "  Class 0: ", np.sum(Ya == 0)
#         print "  Class 1: ", np.sum(Ya == 1)
#         print "  Class 2: ", np.sum(Ya == 2)

        if args.draw_classes_histogram:
            draw_classes_histogram(X, Ya, len(split_points))

        if args.bayes:
            print "Bayes classifier..."
            # Gaussian plug-in classifier
            
            gpi_classifier = GaussianPlugInClassifier(X, Ya, len(split_points))
                
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(gpi_classifier.classify(X_test, Ya_test)[0])
    
        if args.naive:
            print "Naive bayes classifier..."
            # Gaussian naive bayes classifier
            
            naive_classifier = GaussianNaiveClassifier(X, Ya, 
                                                       len(split_points))
            
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(naive_classifier.classify(X_test, Ya_test)[0])
