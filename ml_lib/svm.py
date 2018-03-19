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
    def output(self):
        print "Linear kernel"

    def compute(self, X1, X2):
        ''' Creates a linear kernel matrix between rows of X1 and X2. '''
        return np.dot(X1, X2.T)

class RBFKernel:
    def __init__(self, width):
        self.width = width

    def output(self):
        print "RBF kernel width %s" % (self.width)

    def compute(self, X1, X2):
        ''' Creates a RBF kernel matrix between rows of X1 and X2. '''
        X1 = X1[:, np.newaxis, :]
        X2 = X2[np.newaxis, :, :]
        
        D = np.sum(np.square(X1 - X2), axis=2)
        return np.exp(-1 * D / self.width)
    
    def __eq__(self, other):
        return isinstance(other, RBFKernel) and self.width == other.width

    def __ne__(self, other):
        return not isinstance(other, RBFKernel) or self.width != other.width

class SVM(object):
    '''
    The Support Vector Machine classifier
    '''

    def __init__(self, X, Y=None, lam=None, kernel=None):
        """ Initializes the SVM.
        X and Y form the training data over which to train the model
        lam is the lambda slack weight parameter, small for loose
            classification, or infinity for strict separating hyperplane
        """
        self.X = X
        self.kernel = None

        self.initialize(Y, lam, kernel)
        
    def initialize(self, Y, lam, kernel):
        self.Y = Y
        self.lam = lam
        
        self.alpha = None
        self.w0 = None
        self.first_w0 = None
        
        kernel = kernel or LinearKernel()
        if True:#kernel != self.kernel:
            self.K = None
            self.kernel = kernel
        
        return self

    def learn(self):
        """ Learns the alpha variables of the dual and precomputes w0 """

        print "SVM with lambda = %f" %(self.lam)
        self.kernel.output()

        X = self.X
        n = X.shape[0]
        Y = self.Y * 2 - 1
        Y_ = Y.reshape((n, 1))

#         self.lin_clf = svm.LinearSVC(loss='hinge')
#         self.lin_clf.fit(X, Y)
        
        if self.K is None:
            self.K = self.kernel.compute(X, X)
        K = self.K
        
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

        # verify they're all approx same, and find their average
        numnonzero = nz[0].shape[0]
        numdiff = 0
        others = []#[self.w0]
        #for i in range(al.shape[0]):
        w0_sum = 0
        for iw in range(numnonzero):
            i = nz[0][iw]
            w0_other = (1 - Y[i] * np.inner(self.alpha * Y, K[i])) / Y[i]
            others.append(w0_other)
            w0_diff = abs((w0_other - self.w0) / self.w0)
            if w0_diff > 0.001:
                numdiff += 1
                #print "   different %d th w0: %f" % (i, w0_other)
                #print "      al=%f  mu=%f" % (al[i], mu[i])
            else:
                w0_sum += w0_other
        new_w0 = w0_sum / (numnonzero - numdiff)
        
        print "w0: %f     \tminus avg w0 = %f" %(self.w0, self.w0-new_w0)
        self.first_w0 = self.w0
        self.w0 = new_w0
        
        if numdiff > 0:
            print "Num nonzero = %d/%d. Num different = %d/%d" % (numnonzero, 
                                                                  n, numdiff, n)
            print "  alpha nonzero = %d/%d" % (np.sum(nz_al), n)
            print "     mu nonzero = %d/%d" % (np.sum(nz_mu), n)
        

    def predict(self, X_test):
        """ Predict class for the given data using this SVM classifier.
            Returns a quantity that is positive if class is 1 and that is 
            proportional to the likelihood of it being so.
        """
        if self.alpha is None: self.learn()

        X0 = X_test
        
        X = self.X
        Y = self.Y * 2 - 1

        Kmn = self.kernel.compute(X0, X)
        ret = np.dot(Kmn, self.alpha * Y) + self.w0
        
        return ret

    def classify(self, X_test, Y_test):
        """ Classify the given set using this SVM classifier.
            Returns confusion matrix of test result accuracy.
        """
        pred = self.predict(X_test)
        class_prediction = np.sign(pred)
        class_prediction = ((class_prediction + 1) / 2).astype(int)
         
        c_matrix = util.confusion_matrix(class_prediction, Y_test, 2)
        acc = util.get_accuracy(c_matrix)
        util.report_accuracy(c_matrix, False)

        class_prediction = np.sign(pred + self.first_w0 - self.w0)
        class_prediction = ((class_prediction + 1) / 2).astype(int)
        c_matrix = util.confusion_matrix(class_prediction, Y_test, 2)
        old_acc = util.get_accuracy(c_matrix)
        if acc < old_acc:
            print "\tOld accuracy was better! %f (old) vs %f (avg)" %(old_acc, acc)
        
        return c_matrix
     
