'''
Created on Jan 25, 2018

@author: enerve
'''

from __future__ import division

import numpy as np

class GaussianNaiveClassifier(object):
    '''
    classdocs
    '''

    def __init__(self, X, Y, num_classes=2):
        self.X = X
        self.Y = Y
        self.num_classes = num_classes
    
    # Function that tests data classification for a gaussian naive plug-in
    # classifier and returns confusion matrix
    def classify(self, X_test, Y_test, class_weights=None):
        if class_weights is None:
            class_weights = [1.0 for x in range(self.num_classes)]        

        num = self.Y.shape[0]

        # Precomputed variables
        meanX = []
        varX = []
        prior = []
        pdensity = []

        for c in range(self.num_classes):
            i_c = self.Y == c   # Indices where y==c
            num_c = np.sum(i_c)
            prior_c = num_c / num

            X_c = self.X[i_c]    # All class datapoints
    
            # Compute variance of the feature
            meanX_c = np.mean(X_c, axis=0)
            diffX_c = X_c - meanX_c
            varX_c = (1.0 / num_c) * np.sum(np.square(diffX_c), axis=0)
                
            prior.append(prior_c)
            meanX.append(meanX_c)
            varX.append(varX_c + 0.0000000001)
            pdensity.append([])
                
        c_matrix = np.zeros((self.num_classes, self.num_classes))
        
        # TODO: vectorize
        for i in range(X_test.shape[0]):
            x = X_test[i]
            y = int(Y_test[i])
            max_prob = -1
            for c in range(self.num_classes):
                pYx_c = prior[c] * np.product(
                    (1 / varX[c]) * np.exp(
                        -0.5 * np.square(x - meanX[c]) / varX[c]))
                pdensity[c].append(pYx_c)
                if max_prob < pYx_c * class_weights[c]:
                    max_prob = pYx_c * class_weights[c]
                    class_prediction = c
            c_matrix[y, class_prediction] += 1
                
        return c_matrix, pdensity
    