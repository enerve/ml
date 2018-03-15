'''
Created on Mar 15, 2018

Methods that deal with plumbing between datasets and ML algorithms, e.g.
cross validation and multiclass classification.

@author erw
'''

import numpy as np

import util

def linear_multiclassify(X, Ya, X_test, Ya_test, split_points,
                         create_classifier):
    n = len(split_points)
#     print "Running linear multiclassifier on %s splits -----" %(n)
#     print split_points
    
    Yp = np.zeros((Ya_test.shape[0], n))
    for i, spl in enumerate(split_points):
        if spl==-1: continue
        print "Splitting at %s" % (spl)
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

    for i in range(n-1):
        Yp[:, i] *= Yp[:, i+1]
    Yp[:, n-1] = - Yp[:, n-1] # pretend n+1 col wudve been negative

    Yguess = np.argmin(Yp, axis=1)
    
    c_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c_matrix[i, j] = np.sum(
                np.logical_and((Yguess == j), (Ya_test == i)))
    print 'Overall test acc: %f%%' % util.get_accuracy(c_matrix)
    print c_matrix
    return c_matrix

def onevsall_multiclassify(X, Ya, X_test, Ya_test, n, create_classifier):
#     print "Running linear multiclassifier on %s splits -----" %(n)
#     print split_points
    
    Yp = np.zeros((Ya_test.shape[0], n))
    for i in range(n):
        Yb = np.zeros((Ya.shape[0]))
        Yb[Ya==i] = 1

        classifier = create_classifier(X, Yb)
        print classifier.classify(X, Yb)
        
        Yb_test = np.zeros((Ya_test.shape[0]))
        Yb_test[Ya_test==i] = 1
        print classifier.classify(X_test, Yb_test)

        Yp_i = classifier.predict(X_test)
        Yp[:, i] = Yp_i
        
#     print Yp.shape
#     print Yp
    Yguess = np.argmax(Yp, axis=1)

    c_matrix = util.confusion_matrix(Ya_test, Yguess, n)    

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
    print "Running one vs all multiclassifier on %d classes -----" %(n)
    
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
