'''
Created on Mar 3, 2018

@author: erw
'''
from __future__ import division

import numpy as np
from cvxopt import matrix, solvers

class SVM(object):
    '''
    The Support Vector Machine classifier
    '''

    def __init__(self, X, Y, lam):
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

    def learn(self):
        """ Learn the alpha vector and w0
        """
        X = self.X
        Y = self.Y * 2 - 1
        n = X.shape[0]
        
        Z = Y.reshape((n, 1)) * X
        K = np.dot(Z, Z.T)  # linear SVM

        # Solve quadratic system of the svm dual, i.e. find the alpha vars
        P = matrix(K)
        #print P
        q = matrix(-np.ones(n))
        G = matrix(np.append(-1 * np.eye(n), np.eye(n), axis=0))
        h = matrix(np.append(np.zeros(n), self.lam * np.ones(n)))
#         G = matrix(-1 * np.eye(n))
#         h = matrix(np.zeros(n))
        #print np.array(G).sum()
        A = matrix(np.reshape(Y.astype(float), (1, n)))
        #print np.array(A).sum()
        b = matrix([0.0])
        
        sol = solvers.qp(P, q, G, h, A, b)

        al = np.reshape(np.array(sol['x']), (n))
        al = np.around(al, 5)
        self.alpha = al
        
        # precompute w0
        i = np.nonzero(al)[0][0] # index of first nonzero alpha var
        self.w0 = (1 - np.inner(al, K[i])) / Y[i]

             
    def classify(self, X_test, Y_test):
        """ Classify the given test set using this SVM classifier.
            Returns confusion matrix of test result accuracy.
        """
        if self.alpha is None: self.learn()

        c_matrix = np.asarray([[0, 0],[0,0]])
 
        X0 = X_test
        Y0 = Y_test
        
        X = self.X
        Y = self.Y * 2 - 1

        print self.alpha
        print "w0: %f" %(self.w0)

        Kmn = np.dot(X0, X.T)   # linear SVM
        Y_value = np.dot(Kmn, self.alpha * Y) + self.w0
        
        print Y_value
        
        class_prediction = np.sign(Y_value)
        class_prediction = ((class_prediction + 1) / 2).astype(int)
         
        for i, y in enumerate(Y0):
            c_matrix[y, class_prediction[i]] += 1
         
        print "Accuracy: %f%%" % (100 * (c_matrix[0, 0] + c_matrix[1, 1]) 
                                  / np.sum(c_matrix))
         
        return c_matrix
     
