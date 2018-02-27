'''
@author erwin

'''
from __future__ import division

import numpy as np
import ml_lib.util as util
import argparse


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



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='path to wdbc file')
    parser.add_argument('--test_portion',
                        help='Which portion to use as test set',
                        default=1, type=int)
    parser.add_argument('--draw_classes_data', action='store_true')
    parser.add_argument('--draw_classes_histogram', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--perceptron', action='store_true')
    parser.add_argument('--sklearn_perceptron', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--logistic', action='store_true')
    args = parser.parse_args()
    
    data = np.genfromtxt(args.file, delimiter=",",
                         converters={1: lambda x: 1.0 if x=='M' else 0.0})
    Y = data[:, 1].astype(int)
    X = data[:, features_to_use()]

    X, Y, X_test, Y_test = util.split_into_train_test_sets(X, Y,
                                                           args.test_portion)
    print "--- WDBC dataset ---"
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
        from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
        # Gaussian plug-in classifier
        gpi_classifier = GaussianPlugInClassifier(X, Y, 2)
        # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(
            gpi_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])

        util.draw_ROC_curve(X_test, Y_test, gpi_classifier)
        # util.draw_classes_pdf(X, Y, gpi_classifier, [0.5, 0.5], 3)
    
    if args.naive:
        print "Naive Bayes classifier..."
        from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier
        # Gaussian naive classifier
        gn_classifier = GaussianNaiveClassifier(X, Y, 2)
        # util.report_accuracy(gn_classifier.classify(X, Y, 0.5)[0])
        util.report_accuracy(
            gn_classifier.classify(X_test, Y_test, [0.5, 0.5])[0])
        
        util.draw_ROC_curve(X_test, Y_test, gn_classifier)

    if args.sklearn_perceptron:
        print "Scikit-learn Perceptron..."
        from sklearn.linear_model import Perceptron
        perceptron = Perceptron(tol=None, max_iter=300000)
        perceptron.fit(X, Y)
        print "Mean accuracy: %s%%" %(100 * perceptron.score(X, Y))

    if args.perceptron:
        print "Perceptron..."
        from ml_lib.perceptron import Perceptron
#         perceptron = Perceptron(X, Y, args.stochastic, 1, 300000, 0)
#         print perceptron.classify(X, Y)
#         print perceptron.classify(X_test, Y_test)

#     if args.perceptron:
#         print "Multiclass Perceptron..."
#         from ml_lib.perceptron import Perceptron
#         
        Ya = Y
        Ya_test = Y_test
        split_points = [-1, 0]
        n = len(split_points)
        
        def create_classifier(X, Y):
            perceptron = Perceptron(X, Y, args.stochastic,
                                    1, 30000, 0)
            return perceptron
            
        util.linear_multiclassify(X, Ya, X_test, Ya_test,
                                  split_points, create_classifier)
            

#         Yp = np.zeros((Ya_test.shape[0], n))
#         for i, spl in enumerate(split_points):
#             if spl==-1: continue
#             print "Splitting at %s" % (spl)
#             Yb = np.zeros((Ya.shape[0]))
#             Yb[Ya==i] = 1
#             perceptron = Perceptron(X, Yb, args.stochastic,
#                                     1, 300000, 0)
#             print perceptron.classify(X, Yb)
#              
#             Yb_test = perceptron.predict(X_test)
#             Yp[:, i] = Yb_test
# 
# #         for j in range(n):
# #             if j >= i:
# #                 Yp[:, j] -= Yb_test
# #             else:
# #                 Yp[:, j] += Yb_test
#         Yp[:, 0] = -Yp[:, 1]
#         Yguess = np.argmax(Yp, axis=1)
#         c_matrix = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 c_matrix[i, j] = np.sum(
#                     np.logical_and((Yguess == j), (Ya_test == i)))
#         print c_matrix

    if args.logistic:
        print "Logistic Regression..."
        from ml_lib.logistic import Logistic
        
        
        Ya = Y
        Ya_test = Y_test
        split_points = [-1, 0]
        n = len(split_points)

        def create_classifier(X, Y):
            logistic = Logistic(X, Y, step_size=0.01, max_steps=15000,
                                reg_constant=0.05)
            return logistic
            
        util.linear_multiclassify(X, Ya, X_test, Ya_test,
                                  split_points, create_classifier)

#         logistic = Logistic(X, Y, step_size=0.01, max_steps=15000,
#                             reg_constant=0.05)
#         print logistic.classify(X, Y)
#         logistic.plot_likelihood_train(False)
#         logistic.plot_likelihood_test(X_test, Y_test, True)
# #         print logistic.classify(X_test, Y_test)



#         test_acc = []
#         logistic = None
#         for steps in range(10, 4010, 100):
#             
#             logistic = Logistic(X, Y, step_size=0.01, max_steps=steps,
#                                 reg_constant=0.01)
#             cm = logistic.classify(X_test, Y_test)
#             test_acc.append((cm[0,0] + cm[1,1]) / Y_test.shape[0])
#         lik_plt = logistic.plot_likelihood_train(False)
#         x_range = range(len(test_acc))
#         plt.plot(x_range, test_acc)
#         plt.show()

