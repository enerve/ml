'''
Created on Mar 3, 2018

@author: erw
'''
from __future__ import division

import numpy as np
from cvxopt import matrix, solvers

import ml_lib.util as util
import matplotlib.pyplot as plt
import math

class LinearKernel:
    def __init__(self):
        print "Linear kernel"

    def compute(self, X1, X2):
        ''' Creates a linear kernel matrix between rows of X1 and X2. '''
        return np.dot(X1, X2.T)

class RBFKernel:
    def __init__(self, width):
        print "RBF kernel width %s" % (width)
        self.width = width

    def compute(self, X1, X2):
        ''' Creates a RBF kernel matrix between rows of X1 and X2. '''
        X1 = X1[:, np.newaxis, :]
        X2 = X2[np.newaxis, :, :]
        
        D = np.sum(np.square(X1 - X2), axis=2)
        return np.exp(-1 * D / self.width)

class SVM(object):
    '''
    The Support Vector Machine classifier
    '''

    def __init__(self, X, Y, lam=None, kernel=None):
        """ Initializes the SVM.
        X and Y form the training data over which to train the model
        lam is the lambda slack weight parameter, small for loose
            classification, or infinity for strict separating hyperplane
        """
        self.X = X
        self.Y = Y
        self.lam = lam
        print "SVM with lambda = %f" %(lam)
        
        self.alpha = None
        self.w0 = None
        
        self.kernel = kernel or LinearKernel()

    def learn(self):
        """ Learns the alpha variables of the dual and precomputes w0 """

        X = self.X
        n = X.shape[0]
        Y = self.Y * 2 - 1
        Y_ = Y.reshape((n, 1))

#         self.lin_clf = svm.LinearSVC(loss='hinge')
#         self.lin_clf.fit(X, Y)
        
        K = self.kernel.compute(X, X)
        
        # Solve quadratic system of the svm dual, i.e. find the alpha vars
        P = matrix(Y_ * K * Y_.T)
        q = matrix(-np.ones(n))
        if self.lam: #soft SVM
            G = matrix(np.append(-1 * np.eye(n), np.eye(n), axis=0))
            h = matrix(np.append(np.zeros(n), self.lam * np.ones(n)))
        else:        #hard SVM
            G = matrix(-1 * np.eye(n))
            h = matrix(np.zeros(n))
        A = matrix(Y.reshape((1, n)).astype(float))
        b = matrix([0.0])
        
        print "...Learning..."
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        self.alpha = np.reshape(np.array(sol['x']), (n))
        
        # precompute w0
        
        if self.lam: #soft SVM
            nz_al = (self.alpha / self.lam > 0.01)
            nz_mu = ((self.lam - self.alpha) / self.lam > 0.01)
            nz = np.nonzero(nz_al & nz_mu)
        else:
            nz = np.nonzero(self.alpha) # i.e. alpha > 0
        i = nz[0][0] # index of first such row
        self.w0 = (1 - Y[i] * np.inner(self.alpha * Y, K[i])) / Y[i]
        print "w0: %f" %(self.w0)

        # verify they're all same
        numnonzero = nz[0].shape[0]
        numdiff = 0
        others = []#[self.w0]
        #for i in range(al.shape[0]):
        for iw in range(numnonzero):
            i = nz[0][iw]
            w0_other = (1 - Y[i] * np.inner(self.alpha * Y, K[i])) / Y[i]
            others.append(w0_other)
            w0_diff = abs((w0_other - self.w0) / self.w0)
            if w0_diff > 0.001:
                numdiff += 1
                #print "   different %d th w0: %f" % (i, w0_other)
                #print "      al=%f  mu=%f" % (al[i], mu[i])
        if numdiff > 0:
            print "Num nonzero = %d/%d. Num different = %d/%d" % (numnonzero, 
                                                                  n, numdiff, n)
            print "  alpha nonzero = %d/%d" % (np.nonzero(al)[0].shape[0], n)
            print "     mu nonzero = %d/%d" % (np.nonzero(mu)[0].shape[0], n)
#             print "   round digits = %d" % (round_digits)
        
#         plt.hist((al * mu)[nz], bins=400)
#         plt.show()
#         plt.hist(al[nz], bins=400)
#         plt.show()
#         plt.hist(mu[nz], bins=400)
#         plt.show()
#         plt.hist(others, bins=100)
#         plt.show()
#         plt.hist(al, bins=100)
#         plt.show()

        #for j in range(nz[0].shape[0]):
        #    i = nz[0][j] # index of j-1th nonzero alpha var
        #    w0 = (1 - Y[i] * np.inner(al * Y, K[i])) / Y[i]
        #    print "  potential w0: %f     for alpha[%d] %f  \twhere y=%d  \tdot=%f" %(
        #        w0, i, al[i], Y[i], (np.inner(al * Y, K[i])))
        #    if j>100: break

    def predict(self, X_test):
        """ Predict class for the given data using this SVM classifier.
            Returns a quantity that is positive if class is 1 and that is 
            proportional to the likelihood of it being so.
        """
        if self.alpha is None: self.learn()

#         dec = self.lin_clf.decision_function(X_test)


        X0 = X_test
        
        X = self.X
        Y = self.Y * 2 - 1

        Kmn = self.kernel.compute(X0, X)
        ret = np.dot(Kmn, self.alpha * Y) + self.w0
        
#         df =  (ret - dec)
#         d2 = np.zeros(df.shape[0])
#         d2[ret*dec > 0] = 1
#         print "Diff between Y predictions:"
#         print "  Average : %f off from %f" % (np.average(df), np.average(dec))
#         print "  Ratio   : %f " % (np.average(df / dec))
#         print "  #match: %s / %s" % (np.sum(d2), X_test.shape[0])
        
        return ret

    def classify(self, X_test, Y_test):
        """ Classify the given set using this SVM classifier.
            Returns confusion matrix of test result accuracy.
        """
        class_prediction = np.sign(self.predict(X_test))
        class_prediction = ((class_prediction + 1) / 2).astype(int)
         
        c_matrix = util.confusion_matrix(class_prediction, Y_test, 2)
        util.report_accuracy(c_matrix, False)

        return c_matrix
     
