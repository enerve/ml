'''
Created on Mar 3, 2018

@author: enerve
'''
from __future__ import division
from cvxopt import matrix, solvers
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import sys

import ml_lib.util as util
import logging

class LinearKernel:
    def __str__(self):
        return "Linear kernel"

    def compute(self, X1, X2):
        ''' Creates a linear kernel matrix between rows of X1 and X2. '''
        return np.dot(X1, X2.T)

class RBFKernel:
    def __init__(self, width, vectorize=True):
        self.width = width
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.vectorize = vectorize

    def __str__(self):
        return "RBF kernel width %s" % self.width

    def compute(self, X1, X2):
        ''' Creates a RBF kernel matrix between rows of X1 and X2. '''
        self.logger.debug("RBF")

        if self.vectorize:
            X1 = X1[:, np.newaxis, :]
            X2 = X2[np.newaxis, :, :]
            
            self.logger.debug("RBF: about to Square")
            Q = np.square(X1 - X2)
            self.logger.debug("RBF: Computed square difference")
            D = np.sum(Q, axis=2)
        else:
            D = np.zeros(X1.shape[0], X2.shape[0])
            i = 0
            for x2 in X2:
                Q_ = np.square(X1 - x2)
                self.logger.debug("RBF: Computed square difference")
                D_ = np.sum(Q, axis=1)
                D[i] = D_
                i += 1
                
            
        self.logger.debug("RBF: Summed square difference")
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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

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
        if kernel != self.kernel:
            self.K = None
            self.kernel = kernel
            self.logger.debug("Using kernel: %s" % kernel)
        
        return self

    def learn(self):
        """ Learns the alpha variables of the dual and precomputes w0 """

        self.logger.debug("SVM with lambda = %f", self.lam or 0)
        logging.debug("%s", self.kernel)

        X = self.X
        n = X.shape[0]
        Y = self.Y * 2 - 1
        Y_ = Y.reshape((n, 1))

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
        
        self.logger.info("...Learning...")
        solvers.options['show_progress'] = False
        self.logger.debug("About to solve qp!")
        sol = solvers.qp(P, q, G, h, A, b)
        self.logger.debug("Solved qp...")

        self.alpha = np.reshape(np.array(sol['x']), (n))
        
        # precompute w0
        
        alpha_max = np.max(self.alpha)
        self.logger.debug("Alpha range: zero to %f", alpha_max)
        
        if self.lam: #soft SVM
            nz_al = (self.alpha / alpha_max > 0.01)
            nz_mu = ((self.lam - self.alpha) / alpha_max > 0.01)
            nz = nz_al & nz_mu
        else:
            nz_al = self.alpha / alpha_max > 0.01
            nz = nz_al

        w0_vals = (1 - Y[nz] * np.inner(self.alpha * Y, K[nz])) / Y[nz]
        

        # Average out the w0s, most of which should be identical anyway.           
        self.w0 = np.mean(w0_vals)

        # logging
        logging.debug("w0: %f", self.w0)
        w0_diff = np.abs((w0_vals - self.w0) / self.w0)
        numdiff = np.sum(w0_diff > 0.001)
#         if numdiff > 0:
        logging.debug("Num nonzero = %d/%d. Num different = %d/%d", 
                      np.sum(nz), n, numdiff, n)
        logging.debug("  alpha nonzero = %d/%d", np.sum(nz_al), n)
        if self.lam:
            logging.debug("     mu nonzero = %d/%d", np.sum(nz_mu), n)
        if numdiff > 0:
            logging.debug(self.alpha)
            logging.debug(self.kernel)
        

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

        c_matrix = util.confusion_matrix(Y_test, class_prediction, 2)
        util.report_accuracy(c_matrix, False)

        return c_matrix
     
