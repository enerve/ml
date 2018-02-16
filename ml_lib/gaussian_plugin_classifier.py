'''
Created on Jan 26, 2018

@author: erw
'''

from __future__ import division

import numpy as np
import math

class GaussianPlugInClassifier(object):
    
    def __init__(self, X, Y, num_classes=2):
        self.X = X
        self.Y = Y
        self.num_classes = num_classes
    
    # Function that tests data classification for a gaussian plug-in classifier
    # and returns confusion matrix
    def classify(self, X_test, Y_test, class_weights=None):
        if class_weights is None:
            class_weights = [1.0 for x in range(self.num_classes)]        
        
        num = self.Y.shape[0]

        # Precomputed variables
        meanX = []
        sX_inv = []
        coeff = []
        pdensity = []
        
        for c in range(self.num_classes):        
            i_c = self.Y == c   # Indices where y==c
            num_c = np.sum(i_c)
            prior_c = num_c / num

            X_c = self.X[i_c]    # All class datapoints
            
            # Compute sigma covariance matrix of the gaussian
            meanX_c = np.mean(X_c, axis=0)
            diffX_c = X_c - meanX_c
            sX_c = (1.0 / num_c) * np.dot(diffX_c.T, diffX_c)

            coeff_c = prior_c / math.sqrt(np.linalg.det(sX_c))
            sX_inv_c = np.linalg.inv(sX_c)

            meanX.append(meanX_c) 
            coeff.append(coeff_c)
            sX_inv.append(sX_inv_c)
            pdensity.append([])
            
            # Quadratic Discriminant Analysis
    #         b = np.dot(meanX_1, sX_1_inv) - np.dot(meanX_0, sX_0_inv)
    #         print b.astype(int)
    #         A = sX_0_inv /2 - sX_1_inv / 2
    #         print A.astype(int)
    
    
        c_matrix = np.zeros((self.num_classes, self.num_classes))

        for i in range(X_test.shape[0]):
            y = int(Y_test[i])
            x = X_test[i]
            max_prob = -1
            for c in range(self.num_classes):
                dx_c = x - meanX[c]       # diff
                pYx_c = coeff[c] * np.exp(-0.5 * np.dot(dx_c, np.dot(sX_inv[c], dx_c.T)))
                pdensity[c].append(pYx_c)
                if max_prob < pYx_c * class_weights[c]:
                    max_prob = pYx_c * class_weights[c]
                    class_prediction = c
            
            c_matrix[y, class_prediction] += 1
        
            #     print "Accuracy: %s%%" % ((c_matrix[0, 0] + c_matrix[1, 1]) / np.sum(c_matrix))
        
        return c_matrix, pdensity
    
        
