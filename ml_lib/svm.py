'''
Created on Mar 3, 2018

@author: erw
'''
from __future__ import division

import numpy as np
from cvxopt import matrix, solvers

from sklearn import svm

class LinearKernel:
    def compute(self, X1, X2):
        ''' Creates a linear kernel matrix between rows of X1 and X2. '''
        return np.dot(X1, X2.T)

class RBFKernel:
    def __init__(self, width):
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
        
        self.alpha = None
        self.w0 = None
        
        self.kernel = kernel or LinearKernel()

    def learn(self):
        """ Learns the alpha variables of the dual and precomputes w0 """

        X = self.X
        n = X.shape[0]
        Y = self.Y * 2 - 1
        Y_ = Y.reshape((n, 1))

        self.lin_clf = svm.LinearSVC(loss='hinge')
        self.lin_clf.fit(X, Y)
        
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
        
        sol = solvers.qp(P, q, G, h, A, b)

        al = np.reshape(np.array(sol['x']), (n))
        al = np.around(al, 5)
        self.alpha = al
        
        # precompute w0
        if self.lam: #soft SVM
            mu = self.lam - al
            nza = np.nonzero(al * mu) # i.e. alpha > 0 and mu > 0
        else:
            nza = np.nonzero(al) # i.e. alpha > 0
        i = nza[0][0] # index of first such row
        self.w0 = (1 - Y[i] * np.inner(al * Y, K[i])) / Y[i]
        
        #for j in range(nza[0].shape[0]):
        #    i = nza[0][j] # index of j-1th nonzero alpha var
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

        dec = self.lin_clf.decision_function(X_test)

        print "w0: %f" %(self.w0)

        X0 = X_test
        
        X = self.X
        Y = self.Y * 2 - 1

        Kmn = self.kernel.compute(X0, X)
        ret = np.dot(Kmn, self.alpha * Y) + self.w0
        
        df =  (ret - dec)
        d2 = np.zeros(df.shape[0])
        d2[df>0] = 1
        print "Diff between Y predictions:"
        print "  Average : %f off from %f" % (np.average(df), np.average(dec))
        print "  Ratio   : %f " % (np.average(df / dec))
        print "  #Greater: %s / %s" % (np.sum(d2), X_test.shape[0])
        
        return ret

    def classify(self, X_test, Y_test):
        """ Classify the given test set using this SVM classifier.
            Returns confusion matrix of test result accuracy.
        """
        class_prediction = np.sign(self.predict(X_test))
        class_prediction = ((class_prediction + 1) / 2).astype(int)
         
        c_matrix = np.asarray([[0, 0],[0,0]])
        for i, y in enumerate(Y_test):
            c_matrix[y, class_prediction[i]] += 1
         
        print "Accuracy: %f%%" % (100 * (c_matrix[0, 0] + c_matrix[1, 1]) 
                                  / np.sum(c_matrix))

        return c_matrix
     
