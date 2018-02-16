'''
Created on Jan 26, 2018

@author: erw
'''
from __future__ import division

import numpy as np
import random
import matplotlib.pyplot as plt

class Perceptron(object):
    '''
    classdocs
    '''


    def __init__(self, X, Y, is_stochastic, step_size, max_steps, reg_constant=0):
        self.X = X
        self.Y = Y
        self.is_stochastic = is_stochastic
        if is_stochastic:
            print "Running Stochastic Perceptron..."
        self.step_size = step_size
        self.max_steps = max_steps
        self.reg_constant = reg_constant
        
        self.w = None
        
#         [427, 276, 151, 276, 151, 276, 151, 276, 276, 151, 276, 151, 276, 151, 276, 148, 151, 276, 146, 276, 80, 139, 276, 49, 276, 151, 276, 151, 276, 105, 276, 147, 276, 68, 276, 151, 276, 143, 276, 91, 276, 149, 276, 107, 276, 140, 276, 104, 276, 140, 276, 112, 276, 134, 276, 117, 276, 128, 276, 122, 276, 125, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 275, 125, 274, 124, 274, 124, 272, 124, 272, 124, 271, 124, 270, 123, 267, 123, 267, 123, 262, 119, 257, 122, 257, 119, 255, 119, 253, 118, 244, 118, 250, 118, 247, 117, 244, 114, 238, 112, 234, 112, 228, 111, 228, 108, 211, 102, 188, 92, 154, 77, 119, 57, 94, 41, 68, 41, 64, 40, 65, 40, 64, 40, 65, 40, 64, 40, 63, 40, 64, 40, 64, 40, 64, 40, 64, 40, 64, 40, 63, 38, 56, 36, 57, 38, 60, 38, 59, 39, 61, 38, 57, 36, 57, 38, 59, 37, 57, 36, 57, 37, 60, 38, 57, 36, 57, 37, 60, 38, 57, 36, 57, 38, 60, 38, 57, 36, 57, 38, 58, 38, 57, 38, 60, 38, 56, 36, 59, 38, 56, 37, 60, 38, 56, 37, 55, 37, 59, 37, 65, 37, 61, 37, 65, 37, 61, 37, 66, 37, 61, 37, 65, 37, 61, 37, 65, 37, 61, 37, 65, 37, 61, 37, 65, 37, 62, 37, 62, 37, 65, 37, 61, 37, 63, 36, 60, 36, 60, 36, 61, 36, 60, 36, 61, 37, 60, 36, 61, 36, 60, 36, 60, 36, 61, 36, 60, 36, 60, 35, 61, 36, 60, 36, 60, 35, 60, 37, 68, 40, 69, 41, 71, 42, 71, 42, 71, 42, 71, 41, 71, 42, 71, 42, 71, 42, 71, 40, 69, 42, 71, 41, 70, 42, 71, 42, 71, 41, 70, 42, 71, 42, 74, 43, 74, 43, 74, 42, 72, 43, 73, 43, 74, 42, 74, 43, 74, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 75, 44, 75, 43, 74, 43, 75, 44, 76, 45, 77, 48, 82, 54, 98, 64, 122, 78, 165, 107, 235, 124, 250, 119, 247, 118, 246, 118, 243, 114, 229, 115, 229, 113, 215, 111, 215, 109, 207, 106, 192, 99, 179, 92, 143, 77, 121, 63, 79, 42, 57, 34, 42, 33, 35, 34, 37, 33, 37, 33, 37, 34, 36, 34, 36, 34, 37, 33, 37, 34, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36]
#         [427, 276, 151, 276, 151, 276, 151, 276, 276, 151, 276, 151, 276, 151, 276, 148, 151, 276, 146, 276, 80, 139, 276, 49, 276, 151, 276, 151, 276, 105, 276, 147, 276, 68, 276, 151, 276, 143, 276, 91, 276, 149, 276, 107, 276, 140, 276, 104, 276, 140, 276, 112, 276, 134, 276, 117, 276, 128, 276, 122, 276, 125, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 276, 125, 276, 125, 276, 124, 275, 125, 274, 124, 274, 124, 272, 124, 272, 124, 271, 124, 270, 123, 267, 123, 267, 123, 262, 119, 257, 122, 257, 119, 255, 119, 253, 118, 244, 118, 250, 118, 247, 117, 244, 114, 238, 112, 234, 112, 228, 111, 228, 108, 211, 102, 188, 92, 154, 77, 119, 57, 94, 41, 68, 41, 64, 40, 65, 40, 64, 40, 65, 40, 64, 40, 63, 40, 64, 40, 64, 40, 64, 40, 64, 40, 64, 40, 63, 38, 56, 36, 57, 38, 60, 38, 59, 39, 61, 38, 57, 36, 57, 38, 59, 37, 57, 36, 57, 37, 60, 38, 57, 36, 57, 37, 60, 38, 57, 36, 57, 38, 60, 38, 57, 36, 57, 38, 58, 38, 57, 38, 60, 38, 56, 36, 59, 38, 56, 37, 60, 38, 56, 37, 55, 37, 59, 37, 65, 37, 61, 37, 65, 37, 61, 37, 66, 37, 61, 37, 65, 37, 61, 37, 65, 37, 61, 37, 65, 37, 61, 37, 65, 37, 62, 37, 62, 37, 65, 37, 61, 37, 63, 36, 60, 36, 60, 36, 61, 36, 60, 36, 61, 37, 60, 36, 61, 36, 60, 36, 60, 36, 61, 36, 60, 36, 60, 35, 61, 36, 60, 36, 60, 35, 60, 37, 68, 40, 69, 41, 71, 42, 71, 42, 71, 42, 71, 41, 71, 42, 71, 42, 71, 42, 71, 40, 69, 42, 71, 41, 70, 42, 71, 42, 71, 41, 70, 42, 71, 42, 74, 43, 74, 43, 74, 42, 72, 43, 73, 43, 74, 42, 74, 43, 74, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 74, 43, 75, 43, 74, 43, 74, 43, 75, 44, 75, 43, 74, 43, 75, 44, 76, 45, 77, 48, 82, 54, 98, 64, 122, 78, 165, 107, 235, 124, 250, 119, 247, 118, 246, 118, 243, 114, 229, 115, 229, 113, 215, 111, 215, 109, 207, 106, 192, 99, 179, 92, 143, 77, 121, 63, 79, 42, 57, 34, 42, 33, 35, 34, 37, 33, 37, 33, 37, 34, 36, 34, 36, 34, 37, 33, 37, 34, 35, 35, 35, 35, 35, 35, 35, 35, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 34, 35, 35, 35, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36, 36, 35, 36, 36]

    def learn(self):
        print np.nonzero(self.Y)[0].shape[0]
        print np.nonzero(1 - self.Y)[0].shape[0]

        Xt = np.append(np.ones((self.X.shape[0], 1)), self.X, axis=1)
        Yt = self.Y * 2 - 1

        print Xt.shape
        print Yt.shape
        

#         w = np.random.rand(Xt.shape[1])
        w = np.ones(Xt.shape[1])#####
        lm = []
        lm0 = []
        lm1 = []
        lmv0 = []
        lmv1 = []
        lw = [[] for k in range(len(w))]
        
        for iter in range(self.max_steps):
            M = [] # indices of misclassified datapoints
            c1 = 0
            c0 = 0
            mv0 = 0
            mv1 = 0
            
            P = Yt * np.dot(Xt, w)
            M = np.where(P <= 0)[0]

#             for i, x in enumerate(Xt):
#                 if Yt[i] * np.inner(x, w) <= 0:
#                     M.append(i)

            if len(M) == 0: 
                print "/////\\\\\\ Found linear separable!"
                break

#             for i in M:
#                 x = Xt[i]
#                 if Yt[i] < 0:
#                     c0 += 1
#                 elif Yt[i] > 0:
#                     c1 += 1
#                 if Yt[i] < 0:
#                     mv0 += -np.inner(x, w)
#                 else:
#                     mv1 += -np.inner(x, w)
#                 
# #             lm.append(len(M))
#             lm0.append(c0)
#             lm1.append(c1)
#             lmv0.append(mv0)
#             lmv1.append(mv1)
#             print c0, c1

            if self.is_stochastic:
                # just pick one randomly from M
                M = [M[random.randint(0, len(M)-1)]]

#             grad = np.zeros(Xt.shape[1])
#             for i in M:
#                 grad -= Yt[i] * Xt[i]
                
            grad = - np.sum((Yt[M] * Xt[M].T), axis=1) / len(M)
#             regterm = np.copy(w)
#             regterm[0] = 0

            if self.reg_constant > 0:
                grad += self.reg_constant * w
                
            eta = self.step_size * 10000 / (10000 + iter)
            
            w = w - grad * eta
            
            if iter % 1 == 0:
                for k in range(len(w)):
                    lw[k].append(w[k])
            
                if iter % 10000 == 0:
    #                 sys.stdout.write('.')
    #                 print grad.astype(int)
                    print "Iter %s:\t" %(iter), w[0], w[1], w[2]
        
        print "Iterations: %s" %(iter)
        print lm
        print lm0
        print lm1
#         print [int(l) for l in lmv]

        x_range = range(len(lw[0]))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
#         ax1.plot(x_range, lmv0, 'r-')
#         ax1.plot(x_range, lmv1, 'g-')

        
        for j, lwn in enumerate(lw):
            if j % 3 >= 2:
                a = w[j]
                ax1.plot(x_range, [(x-a) for x in lwn], label=str(j))

        plt.xlabel("Iteration")
        plt.ylabel("Feature weight")
        plt.show()
        
        print w.astype(int)
        print np.array2string(w, precision=2, separator=',')
        
        self.w = w
    
    def classify(self, X_test, Y_test):
        if self.w is None: self.learn()
        
        c_matrix = np.asarray([[0, 0],[0,0]])

        Xt = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
#         Xt = X_test
        Yt = Y_test
        
        class_prediction = np.sign(np.dot(Xt, self.w))
        class_prediction = ((class_prediction + 1) / 2).astype(int)
        
        for i, y in enumerate(Yt):
            c_matrix[y, class_prediction[i]] += 1
        
        print "Accuracy: %f%%" % (100 * (c_matrix[0, 0] + c_matrix[1, 1]) / np.sum(c_matrix))
        return c_matrix