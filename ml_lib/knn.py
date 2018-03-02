'''
Created on Feb 28, 2018

@author: erw
'''

import numpy as np

class KNN(object):
    '''
    k-Nearest Neighbor algorithm for classification
    '''


    def __init__(self, X, Y, k, num_classes=2):
        '''
        Constructor
        '''
        self.X = X
        self.Y = Y
        self.k = k
        self.num_classes = num_classes
        
    def classify(self, X_test, Y_test):
        
        X1 = self.X[:, np.newaxis, :]
        X2 = X_test[np.newaxis, :, :]

        D = np.square(X1 - X2)  # (train x / test x / features)
        D = np.sum(D, axis=2)   # (train x / test x)
        
        idx = np.argpartition(D, self.k-1, axis=0)
        
        C = self.Y[idx]
#         freq = np.bincount(C[0:self.k-1, :])
        C = C[0:self.k, :]
                
        Yo = []
        for ynn in C.T:
            freq = np.bincount(ynn)
            Yo.append(np.argmax(freq))
            
        Yo = np.asarray(Yo)
        
        c_matrix = np.zeros((self.num_classes, self.num_classes))
        for i, y in enumerate(Y_test):
            c_matrix[y, Yo[i]] += 1

        return c_matrix