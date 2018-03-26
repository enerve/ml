'''
Created on Mar 9, 2018

@author: enerve
'''
from __future__ import division
import logging
import numpy as np

from sklearn import svm

import ml_lib.util as util

class SVMSkSVC(object):
    '''
    Wrapper around sklearn's SVC implementation
    '''

    def __init__(self, X, Y, lam, b, kernel='rbf'):
        """ Initializes the SVM.
        X and Y form the training data over which to train the model
        lam is the lambda slack weight parameter, small for loose
            classification, or infinity for strict separating hyperplane
        """
        self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging.INFO)

        self.X = X
        self.Y = Y
        self.lam = lam
        self.b = b
        self.kernel = kernel
        
        self.clf = None

    def learn(self):
        """ Learns the alpha variables of the dual and precomputes w0 """

        self.logger.info("Sklearn SVM with lambda = %f, b = %f and kernel %s",
            self.lam, self.b, self.kernel)

        X = self.X
        n = X.shape[0]
        Y = self.Y * 2 - 1

        self.clf = svm.SVC(C=self.lam, gamma=1/self.b)
        self.clf.fit(X, Y)  
        

    def predict(self, X_test):
        """ Predict class for the given data using this SVM classifier.
            Returns a quantity that is positive if class is 1 and that is 
            proportional to the likelihood of it being so.
        """
        if self.clf is None: self.learn()

        dec = self.clf.decision_function(X_test)

        return dec

    def classify(self, X_test, Y_test):
        """ Classify the given test set using this SVM classifier.
            Returns confusion matrix of test result accuracy.
        """
        class_prediction = np.sign(self.predict(X_test))
        class_prediction = ((class_prediction + 1) / 2).astype(int)
         
        c_matrix = util.confusion_matrix(Y_test, class_prediction, 2)
        util.report_accuracy(c_matrix, False)

        return c_matrix
     
