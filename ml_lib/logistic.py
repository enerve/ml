'''
Created on Feb 20, 2018

@author: erw
'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import util

def sigmoid(A):
    e = np.exp(-A)
    return 1 / (1 + e)

class Logistic(object):
    '''
    The Logistic Regression classifier, intended to be run on a dataset of two
    classes.
    '''

    def __init__(self, X, Y, step_size, max_steps, reg_constant=0):
        """ Initializes the Logistic regression classifier.
        X and Y is the training data over which to learn the hyperplane
        step_size is the learning rate to be used.
        max_steps is the maximum number of iterations to use
        reg_constant is the regularization multiplier to be used.
        """
        self.X = X
        self.Y = Y
        self.step_size = step_size
        self.max_steps = max_steps
        self.reg_constant = reg_constant
        
        self.w = None


    def learn(self):
        """ Learn the logistic regression parameters w for the training data
        """
        Xt = np.append(np.ones((self.X.shape[0], 1)), self.X, axis=1)
        Yt = self.Y * 2 - 1

        w = np.ones(Xt.shape[1])    # avoiding random init, for debugging
        self.stats_lik = []
        self.stats_w = []
        
#         lw = [[] for k in range(len(w))]
        for iter in range(self.max_steps):
#             print w
            P = sigmoid(Yt * np.dot(Xt, w))
            grad = np.sum(Yt * (1-P) * Xt.T, axis=1) - self.reg_constant * w
            eta = self.step_size# * 10000 / (10000 + iter)
            w = w + grad * eta
            
            if iter % 10 == 0:
                likelihood = np.mean(np.log(P))
                self.stats_lik.append(likelihood)
                self.stats_w.append(w)
#                 for k in range(len(w)):
#                     lw[k].append(w[k])
            
                if iter % 10000 == 0:
                    print "Iter %s:\t" %(iter), w[0], w[1], w[2]
        
        print "Iterations: %s" %(iter)

    

#         x_range = range(len(lw[0]))
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
# 
#         
#         for j, lwn in enumerate(lw):
#             if j % 3 >= 2:  # plot an arbitrary subset of features
#                 a = w[j]
#                 ax1.plot(x_range, [(x-a) for x in lwn], label=str(j))
# 
#         plt.xlabel("Iteration")
#         plt.ylabel("Feature weight")
#         plt.show()
        
#         print np.array2string(w, precision=2, separator=',')
        
        self.w = w
        
    def plot_likelihood_train(self, show_plot=True, prefix=None):
        if self.w is None: self.learn()

        x_range = range(len(self.stats_lik))
        plt.plot(x_range, self.stats_lik, 'b-')
        if show_plot:
            plt.xlabel("Iteration")
            plt.ylabel("Likelihood")
            fname = self.prefix() + \
                + ("%s_"%prefix if prefix else '') \
                + 'train.png'
            plt.savefig(fname, bbox_inches='tight')
            plt.show()

    def plot_likelihood_improvement_train(self):
        if self.w is None: self.learn()

        x_range = range(len(self.stats_lik))

        lik = self.stats_lik
        impr = (np.asarray(lik[1:]) - np.asarray(lik[0:-1])).tolist()

#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
        plt.plot(x_range[3:], impr[2:])
        plt.xlabel("Iteration")
        plt.ylabel("Likelihood improvement")
        plt.show()

    def plot_likelihood_test(self, X_test, Y_test, show_plot=True,
                             prefix=None):
        if self.w is None: self.learn()

        Xt = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
        Yt = Y_test * 2 - 1

        lik = []
        for w in self.stats_w:
            P = sigmoid(Yt * np.dot(Xt, w))
            lik.append(np.mean(np.log(P)))
        
        x_range = range(len(lik))
        plt.plot(x_range, lik, 'r:')
        if show_plot:
            plt.xlabel("Iteration")
            plt.ylabel("Likelihood")
            fname = self.prefix() \
                + ("%s_"%prefix if prefix else '') \
                + 'ttest.png'
            plt.savefig(fname, bbox_inches='tight')
            plt.show()

    def classify(self, X_test, Y_test):
        """ Classify the given test set using this logistic classifier.
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
        
        print "Accuracy: %f%%" % (100 * (c_matrix[0, 0] + c_matrix[1, 1]) 
                                  / np.sum(c_matrix))
        
        return c_matrix
    
    def prefix(self):
        return util.prefix() +\
            "a%s_"%self.step_size + \
            "r%s_"%self.reg_constant + \
            "s%s_"%self.max_steps

    def predict(self, X_test):
        """ Predict class for the given data, and return for each, a quantity 
            that is positive if class is 1 and that is proportional to the
            likelihood of it being so.
        """
        if self.w is None: self.learn()
        
        Xt = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
        return np.dot(Xt, self.w)
