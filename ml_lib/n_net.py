'''
Created on May 17, 2018

Implementation of Neural Networks

@author: enerve
'''

from __future__ import division
import logging
import numpy as np
import ml_lib.util as util


class NNet():
    '''
    A neural network classifier.
    '''

    def __init__(self, X, Y, hidden_layer_sizes, learning_rate,
                 reg_constant):
        '''
        Constructor
        X The training input data, whose axis=1 is the size of input layer
        Y The training output values, whose axis=1 is the size of output layer
        hidden_layer_sizes is a list of sizes of hidden layers, excluding
            input and output layers, starting near input layer.
        learning_rate Learning rate parameter for gradient descent
        reg_constant Regularization parameter for the weight cost
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.lam = reg_constant

        self.layer_sizes = [X.shape[1]]
        self.layer_sizes.extend(hidden_layer_sizes)
        self.layer_sizes.append(Y.shape[1])
        S = self.layer_sizes
        L = len(S)
        assert X.shape[1] == S[0], \
            "First layer size should match #Features of input X"
        assert Y.shape[1] == S[-1], \
            "Last layer size should match #classes in output Y"
        
        self.restart()

    def restart(self):
        # Initialize weights
        EPSILON = 0.01
        S = self.layer_sizes
        L = len(S)

        self.thetas = []
        for i in range(L-1):
            theta = 2 * EPSILON * np.random.rand(S[i+1], S[i] + 1) - EPSILON
            self.thetas.append(theta)
        
        self.iteration = 0

        self.logger.debug("Starting NNet with layers %s, Epsilon=%0.4f, " +
                          "LR=%0.2f, lam=%0.4f",
                          S, EPSILON, self.learning_rate, self.lam)
        
    def train(self, num_iterations):
        ''' Train the neural network for the given number of iterations
        '''
        L = len(self.layer_sizes)
        X = self.X
        Y = self.Y
        m = X.shape[0] # number of data points
                
        Ts = self.thetas

        for iter in range(num_iterations):
            self.iteration += 1
                        
            A_s = []
            # Forward propogation
            #self.logger.debug("Iteration %d. Forward prop", iter)
            A = X.T
            for i in range(L-1):
                A_ = np.append(np.ones((1, A.shape[1])), A, axis=0)
                A_s.append(A_)
                Z = np.dot(Ts[i], A_)
                A = self.sigmoid(Z)

            if iter == num_iterations - 1:
                # Cost
                error_cost = - np.sum(Y.T * np.log(A) + 
                                      #(1 - Y.T) * (np.log(1.0000001 - A))) / m
                                      (1 - Y.T) * (np.log(1 - A))) / m
                reg_cost = 0
                for i in range(L-1): # regularization
                    reg_cost +=  np.sum(Ts[i] * Ts[i]) * self.lam / (2 * m)
                #cost = error_cost + reg_cost
                
                # monitor activation saturation
                num_saturated = 0
                num_activations = 0
                for A_ in A_s:
                    #sum_activations += np.sum(np.absolute(A))
                    num_saturated += np.sum(np.logical_or(A_ > 0.999, A_ < 0.001))
                    num_activations += A_.shape[1] * A_.shape[0]
                perc_saturated = 100 * num_saturated / num_activations
            
                tr_accuracy = util.get_accuracy(
                    self.classify(X, Y, should_report=False))

                
            # Back propagation
            # Precompute delta D for last (output) layer
            D = A - Y.T # initialize to error cost of output
            for i in reversed(range(L-1)):
                # Compute cost gradient for theta = D*A matrix (averaged over
                #   all training examples)
                Tcost_grad = np.dot(D, A_s[i].T)
                # Regularization
                T_reg = np.copy(Ts[i])
                T_reg[:, 0] = 0
                Tcost_grad += self.lam * T_reg
                Tcost_grad /= m
                
                # Precompute delta D for next loop iteration (earlier layer)
                D = np.dot(Ts[i].T, D) * A_s[i] * (1 - A_s[i])
                D = D[1:, :] # Because bias value (=1) is not to be modified
                
                # Update theta for the current layer
                Ts[i] -= Tcost_grad * self.learning_rate

            # TODO: gradient checking

        #self.thetas = Ts
        #self.logger.debug("Thetas: %s", self.thetas)
        
        self.stat_J_e = error_cost
        self.stat_J_r = reg_cost
        self.stat_perc_saturated = perc_saturated
        self.stat_acc = tr_accuracy

    def get_stats(self):
        return (self.iteration, self.stat_J_e, self.stat_J_r, 
                self.stat_perc_saturated, self.stat_acc)


    def predict(self, X_test):
        L = len(self.layer_sizes)

        A = X_test.T
        for i in range(L-1):
            A_ = np.append(np.ones((1, A.shape[1])), A, axis=0)
            Z = np.dot(self.thetas[i], A_)
            A = self.sigmoid(Z)
            
        return A.T
        
    def classify(self, X_test, Y_test, should_report=True):
        Y_p = self.predict(X_test)
        Y_guess_ind = np.argmax(Y_p, axis=1)
        Y_guess_row = np.array(range(Y_p.shape[0]))
        Y_guess = np.zeros(Y_p.shape)
        Y_guess[Y_guess_row, Y_guess_ind] = 1

        n = Y_test.shape[1]
        
        c_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                c_matrix[i, j] = np.sum(
                    np.logical_and(Y_test[:, i], Y_guess[:, j]))

        if should_report:
            util.report_accuracy(c_matrix)
        return c_matrix
        
    def sigmoid(self, Z):
        return 1.0 / (1 + np.exp(-Z))
    
    def prefix(self):
        return \
            "L%s_"%self.layer_sizes + \
            "r%s_"%self.lam + \
            "l%s_"%self.learning_rate
    
    
class NNetClassifier():
    ''' A label-based classifier that uses a neural network.
        A wrapper around a NNet object that propely handles the output labels
            to be mapped to output units.
    '''
    
    def __init__(self, X, Y, hidden_layer_sizes, max_steps, learning_rate,
                 reg_constant, num_classes=2):
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        Y_spread = self.spread(Y)
        self.logger.debug("3==>%s", Y_spread.shape)
        self.nnet = NNet(X, Y_spread, hidden_layer_sizes, max_steps,
                         learning_rate, reg_constant)

    def spread(self, Y):
        Y_row = np.array(range(Y.shape[0]))
        Y_ = np.zeros((Y.shape[0], self.num_classes), dtype=np.int8)
        Y_[Y_row, Y] = 1
        return Y_

    def train(self):
        return self.nnet.train()
    
    def classify(self, X_test, Y_test):
        Y_test_spread = self.spread(Y_test)
        return self.nnet.classify(X_test, Y_test_spread)

    def predict(self, X_test):
        return self.nnet.predict(X_test)[:, 1] # only predictions for "1"

