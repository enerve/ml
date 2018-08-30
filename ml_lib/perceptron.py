'''
Created on Jan 26, 2018

@author: enerve
'''
from __future__ import division

import numpy as np
import random
import matplotlib.pyplot as plt

class Perceptron(object):
    '''
    The Perceptron classifier, intended to be run on a dataset of two classes
    that are linearly separable.
    '''

    def __init__(self, X, Y, is_stochastic, step_size, max_steps,
                 reg_constant=0):
        """ Initializes the Perceptron classifier.
        X and Y is the training data over which to learn the hyperplane
        If is_stochastic is True then the perceptron gradient steps will be
        stochastic not batch.
        step_size is the learning rate to be used.
        max_steps is the maximum number of iterations to use before giving up
        reg_constant is the regularization multiplier to be used.
        """
        self.X = X
        self.Y = Y
        self.is_stochastic = is_stochastic
        if is_stochastic:
            print("Running Stochastic Perceptron...")
        self.step_size = step_size
        self.max_steps = max_steps
        self.reg_constant = reg_constant
        
        self.w = None

    def learn(self):
        """ Learn the separating hyperplane on the training data
        """
        Xt = np.append(np.ones((self.X.shape[0], 1)), self.X, axis=1)
        Yt = self.Y * 2 - 1

        w = np.ones(Xt.shape[1])    # avoiding random init, for debugging
        lw = [[] for k in range(len(w))]
        
        for iter in range(self.max_steps):
            P = Yt * np.dot(Xt, w)
            M = np.where(P <= 0)[0]  # indices of misclassified datapoints

            if len(M) == 0: 
                print("Found linearly separable hyperplane!")
                break

            if self.is_stochastic:
                # just pick one randomly from M
                M = [M[random.randint(0, len(M)-1)]]

            grad = -1 * np.sum((Yt[M] * Xt[M].T), axis=1) / len(M)

            if self.reg_constant > 0:
                grad += self.reg_constant * w
                
            eta = self.step_size * 10000 / (10000 + iter)
            
            w = w - grad * eta
            
            if iter % 100 == 0:
                for k in range(len(w)):
                    lw[k].append(w[k])
            
                if iter % 1000 == 0:
                    print("Iter %s:\t %f %f %f" %(iter, w[0], w[1], w[2]))
        
        print("Iterations: %s" %(iter))

#         x_range = range(len(lw[0]))
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)        
#         for j, lwn in enumerate(lw):
#             if j % 3 >= 2:  # plot an arbitrary subset of features
#                 a = w[j]
#                 ax1.plot(x_range, [(x-a) for x in lwn], label=str(j))
# 
#         plt.xlabel("Iteration")
#         plt.ylabel("Feature weight")
#         plt.show()
        
        print("%s" % np.array2string(w, precision=2, separator=','))
        
        self.w = w
    
    def classify(self, X_test, Y_test):
        """ Classify the given test set using the learned perceptron (and
            learning the perceptron to being with if not already done so).
            Returns confusion matrix of test result accuracy.
        """
        if self.w is None: self.learn()
        
        c_matrix = np.asarray([[0, 0],[0,0]])

        Xt = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
        Yt = Y_test
        
        class_prediction = np.sign(np.dot(Xt, self.w))
        class_prediction = ((class_prediction + 1) / 2).astype(int)
        
        for i, y in enumerate(Yt):
            c_matrix[y, class_prediction[i]] += 1
        
#         print "Accuracy: %f%%" % (100 * (c_matrix[0, 0] + c_matrix[1, 1]) 
#                                   / np.sum(c_matrix))
        return c_matrix
    
    def predict(self, X_test):
        """ Predict class for the given data, and return for each, a quantity 
            that is positive if class is 1 and that is proportional to the
            likelihood of it being so.
        """
        if self.w is None: self.learn()
        
        Xt = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
        return np.dot(Xt, self.w)
