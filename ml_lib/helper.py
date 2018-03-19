'''
Created on Mar 15, 2018

Methods that deal with plumbing between datasets and ML algorithms, e.g.
cross validation and multiclass classification.

@author erw
'''

import numpy as np

import ml_lib.util as util

def linear_multiclassify(X, Ya, X_test, Ya_test, num_classes,
                         create_classifier):
    
    Yp = np.zeros((Ya_test.shape[0], num_classes))
    for i in range(1, num_classes):
        Yb = np.zeros((Ya.shape[0]))
        Yb[Ya>=i] = 1
        classifier = create_classifier(X, Yb)
        print classifier.classify(X, Yb)

        Yb_test = np.zeros((Ya_test.shape[0]))
        Yb_test[Ya_test>=i] = 1
        util.report_accuracy(classifier.classify(X_test, Yb_test))
        Yp_i = classifier.predict(X_test)
        Yp[:, i] = Yp_i
        
        ### rmoeve?
#         classifier.plot_likelihood_train(False, str(i))
#         Yb_test = np.zeros(Ya_test.shape[0])
#         Yb_test[Ya_test>=i] = 1
#         classifier.plot_likelihood_test(X_test, Yb_test, True, str(i))
        
    Yp[:, 0] = 1    #it's gotta be positive by definition.

    for i in range(num_classes-1):
        Yp[:, i] *= Yp[:, i+1]
    Yp[:, num_classes-1] = - Yp[:, num_classes-1] # pretend n+1 col wudve been negative

    Y_guess = np.argmin(Yp, axis=1)
    
    c_matrix = util.confusion_matrix(Ya_test, Y_guess, num_classes)
    print 'Overall test acc: %f%%' % util.get_accuracy(c_matrix)
    print c_matrix
    return c_matrix

def onevsone_multiclassify(X, Y, X_test, Y_test, num_classes,
                           create_classifier):

    Yp = np.zeros((Y_test.shape[0], num_classes, num_classes))
    for i in range(num_classes - 1):
        for j in range(i+1, num_classes):
            print "Classifying %s vs %s" % (i, j)
            id_ij = np.logical_or(Y == i, Y == j)
            n_ij = np.sum(id_ij)
            Xa = X[id_ij]
            Ya = Y[id_ij]
            Yb = np.zeros(n_ij)
            Yb[Ya==i] = 1

            classifier = create_classifier(Xa, Yb)
            Yp[:, i, j] = classifier.predict(X_test)
            Yp[:, j, i] = -Yp[:, i, j]

    Yp = np.sum(Yp, axis=2)
    Y_guess = np.argmax(Yp, axis=1)
    
    c_matrix = util.confusion_matrix(Y_test, Y_guess, num_classes)    

    print 'Overall test acc: %f%%' % util.get_accuracy(c_matrix)
    print c_matrix
    print "......."
    print "......."
    print "......."
    
    return c_matrix


def onevsall_multiclassify(X, Y, X_test, Y_test, n, create_classifier):
    print "Running one vs all multiclassifier on %d classes -----" %(n)

    Yp = np.zeros((Y_test.shape[0], n))
    for i in range(n):
        Yb = np.zeros((Y.shape[0]))
        Yb[Y==i] = 1
        Yb_test = np.zeros((Y_test.shape[0]))
        Yb_test[Y_test==i] = 1

        classifier = create_classifier(X, Yb)
        #print classifier.classify(X, Yb)
        
        print classifier.classify(X_test, Yb_test)

        Yp_i = classifier.predict(X_test)
        Yp[:, i] = Yp_i
        
#     print Yp.shape
#     print Yp
    Yguess = np.argmax(Yp, axis=1)

    c_matrix = util.confusion_matrix(Y_test, Yguess, n)    

    print 'Overall test acc: %f%%' % util.get_accuracy(c_matrix)
    print c_matrix
    print "......."
    print "......."
    print "......."
    
    return c_matrix

def validate_for_best(acc_fn, var_list, depth):
    if var_list is None:
        return acc_fn()
    
    best_accuracy = -1
    acc_list = []
    best_classifier = None
    # Try all variations on the variables
    for var in var_list:
        if depth == 0:
            print "============== var1 = %f ===========" %(var)
        elif depth == 1:
            print "-------------- var2 = %f -----------" %(var)
        (classifier_v, best_acc_v, acc_list_v) = acc_fn(var)
        acc_list.append(acc_list_v)
        if best_acc_v > best_accuracy:
            best_accuracy = best_acc_v
            best_classifier = classifier_v
            #print "  *** Found better with %f%%" % acc_v
            
    return (best_classifier, best_accuracy, acc_list)

def onevsall_multiclassify_validation(X, Y, X_val, Y_val, n,
                                      create_classifier_validated,
                                      var1_list, var2_list):
    print "Running one vs all multiclassifier " + \
        "validation on %d classes -----" %(n)
    
    Yp = np.zeros((Y_val.shape[0], n))
    
    acc_list = []
    for i in range(n):
        Yb = np.zeros((Y.shape[0]))
        Yb[Y==i] = 1
        Yb_val = np.zeros((Y_val.shape[0]))
        Yb_val[Y_val==i] = 1
        
        def classifier_and_accuracy(var1, var2):
            classifier = create_classifier_validated(var1, var2, X, Yb)
            cm_v = classifier.classify(X_val, Yb_val)
            acc = util.get_accuracy(cm_v)
            return (classifier, acc, acc)

        best_classifier, best_acc_v, acc_list_v = validate_for_best(
            lambda var1: validate_for_best(
                lambda var2, var1=var1:
                    classifier_and_accuracy(var1=var1, var2=var2),
                var2_list, 1),
            var1_list, 0)
        acc_list.append(acc_list_v)

        print
        print "============== DONE class %d ===============" % (i)
        print
        print "Best:"
        print best_classifier.classify(X_val, Yb_val)
        
        Yp_i = best_classifier.predict(X_val)
        Yp[:, i] = Yp_i
        
    print "============== DONE ALL ==============="
    print
    Yguess = np.argmax(Yp, axis=1)

    c_matrix = util.confusion_matrix(Y_val, Yguess, n)

    print 'Overall test acc: %f%%' % util.get_accuracy(c_matrix)
    print c_matrix
    print "......."
    print "......."
    print "......."
    
    return c_matrix, acc_list
