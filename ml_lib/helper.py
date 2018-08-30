'''
Created on Mar 15, 2018

Methods that deal with plumbing between datasets and ML algorithms, e.g.
cross validation and multiclass classification.

@author enerve
'''

import logging
import numpy as np

import ml_lib.util as util

logger = logging.getLogger(__name__)

def init_logger():
    #logger.setLevel(logging.INFO)
    pass
    

def linear_multiclassify(X, Ya, X_test, Ya_test, num_classes,
                         create_classifier):
    
    Yp = np.zeros((Ya_test.shape[0], num_classes))
    for i in range(1, num_classes):
        Yb = np.zeros((Ya.shape[0]))
        Yb[Ya>=i] = 1
        classifier = create_classifier(X, Yb)
        logging.debug(classifier.classify(X, Yb))

        Yb_test = np.zeros((Ya_test.shape[0]))
        Yb_test[Ya_test>=i] = 1
        util.report_accuracy(classifier.classify(X_test, Yb_test))
        Yp_i = classifier.predict(X_test)
        Yp[:, i] = Yp_i
        
    Yp[:, 0] = 1    #it's gotta be positive by definition.

    for i in range(num_classes-1):
        Yp[:, i] *= Yp[:, i+1]
    Yp[:, num_classes-1] = - Yp[:, num_classes-1] # pretend n+1 col wudve been negative

    Y_guess = np.argmin(Yp, axis=1)
    
    c_matrix = util.confusion_matrix(Ya_test, Y_guess, num_classes)
    logging.info('Overall test acc: %f%%', util.get_accuracy(c_matrix))
    logging.debug('%s', c_matrix)
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
            logging.info("============== var1 = %f ===========", var)
        elif depth == 1:
            logging.info("-------------- var2 = %f -----------", var)
        (classifier_v, best_acc_v, acc_list_v) = acc_fn(var)
        acc_list.append(acc_list_v)
        if best_acc_v > best_accuracy:
            best_accuracy = best_acc_v
            best_classifier = classifier_v
            
    return (best_classifier, best_accuracy, acc_list)

def classifier_helper(X, Y, X_val, Y_val, create_classifier_validated):
    classifier = create_classifier_validated(X, Y)
    cm_v = classifier.classify(X_val, Y_val)
    acc = util.get_accuracy(cm_v)
    return (classifier, acc, acc)

def class_validation_helper_one_var(var_list, X, Y, X_val, Y_val,
                                    create_classifier_validated):
    def classifier_and_accuracy(var):
        classifier = create_classifier_validated(X, Y, var)
        cm_v = classifier.classify(X_val, Y_val)
        acc = util.get_accuracy(cm_v)
        return (classifier, acc, acc)
    
    return validate_for_best(
            lambda var: classifier_and_accuracy(var=var),
            var_list, 0)

def class_validation_helper_two_var(var1_list, var2_list, X, Y, X_val, Y_val,
                                    create_classifier_validated):
    def classifier_and_accuracy(var1, var2):
        classifier = create_classifier_validated(X, Y, var1, var2)
        cm_v = classifier.classify(X_val, Y_val)
        acc = util.get_accuracy(cm_v)
        return (classifier, acc, acc)

    return validate_for_best(
        lambda var1: validate_for_best(
            lambda var2, var1=var1:
                classifier_and_accuracy(var1=var1, var2=var2),
            var2_list, 1),
        var1_list, 0)

def classifier_helper_for(var_lists, X, Y, X_val, Y_val, num_classes,
                          create_classifier):
    # TODO: remove "num_classes"
    if len(var_lists) == 0:
        c_helper = lambda X, Y, X_val, Y_val, info: \
            classifier_helper(X, Y, X_val, Y_val, create_classifier)
    elif len(var_lists) == 1:
        c_helper = lambda X, Y, X_val, Y_val, info: \
            class_validation_helper_one_var(
                var_lists[0], X, Y, X_val, Y_val, create_classifier)
    elif len(var_lists) == 2:
        c_helper = \
            lambda X, Y, X_val, Y_val, info: \
                class_validation_helper_two_var(
                    var_lists[0], var_lists[1], X, Y, X_val, Y_val,
                    create_classifier)
    return c_helper

def classify(var_lists, X, Y, X_val, Y_val, create_classifier):

    return classify_class(X, Y, X_val, Y_val,
                          classifier_helper_for(var_lists, X, Y, X_val, Y_val,
                                                2, create_classifier))

def classify_class(X, Y, X_val, Y_val, class_validation_helper):
    logging.info("Running classifier")
     
    best_classifier, best_acc_v, acc_list_v = \
        class_validation_helper(X, Y, X_val, Y_val, info=None)

    logging.debug("")
    c_matrix = best_classifier.classify(X_val, Y_val)
 
    logging.info('Validation acc: %f%%', util.get_accuracy(c_matrix))
    logging.debug('%s', c_matrix)
    logging.debug(".......")
     
    return c_matrix, acc_list_v

def classify_one_vs_all(var_lists, X, Y, X_val, Y_val, num_classes,
                        create_classifier):
    c_helper = classifier_helper_for( \
        var_lists, X, Y, X_val, Y_val, num_classes,
                        create_classifier)

    return one_vs_all_multiclassify(X, Y, X_val, Y_val, num_classes,
                                    c_helper)

def one_vs_all_multiclassify(X, Y, X_val, Y_val, num_classes,
                             c_helper):
    logging.info("Running one vs all multiclassifier " + \
                 "validation on %d classes -----", num_classes)
    
    Yp = np.zeros((Y_val.shape[0], num_classes))
    
    acc_list = []
    for i in range(num_classes):
        Yb = np.zeros((Y.shape[0]))
        Yb[Y==i] = 1
        Yb_val = np.zeros((Y_val.shape[0]))
        Yb_val[Y_val==i] = 1
        
        classifier, acc_v, acc_list_v = c_helper(X, Yb, X_val, Yb_val, (i,))
        acc_list.append(acc_list_v)

        logging.debug("")
        logging.info("============== DONE class %d ===============", i)
        logging.debug("")
        cm_b = classifier.classify(X_val, Yb_val)
        logging.debug("%s", cm_b)
        
        # Record test prediction for later comparison
        Yp_i = classifier.predict(X_val)
        Yp[:, i] = Yp_i
        
    logging.info("============== DONE ALL ===============")
    logging.debug("")
    
    Yguess = np.argmax(Yp, axis=1)

    c_matrix = util.confusion_matrix(Y_val, Yguess, num_classes)

    logging.info('Overall test acc: %f%%', util.get_accuracy(c_matrix))
    logging.debug('%s', c_matrix)
    logging.debug(".......")
    
    return c_matrix, acc_list

def one_vs_one_partition(X, Y, i, j):
    id_ij = np.logical_or(Y == i, Y == j)
    n_ij = np.sum(id_ij)
    Xa = X[id_ij]
    Ya = Y[id_ij]
    Yb = np.zeros(n_ij, dtype=np.int8)
    Yb[Ya==i] = 1
    return (Xa, Yb)

def classify_one_vs_one(var_lists, X, Y, X_val, Y_val, num_classes,
                        create_classifier):
    c_helper = classifier_helper_for( \
                        var_lists, X, Y, X_val, Y_val, num_classes,
                        create_classifier)

    return one_vs_one_multiclassify(X, Y, X_val, Y_val, num_classes,
                                    c_helper)

def one_vs_one_multiclassify(X, Y, X_val, Y_val, num_classes,
                             c_helper):
    logging.info("Running one vs one multiclassifier " + \
                 "on %d classes -----", num_classes)
    
    Yp = np.zeros((Y_val.shape[0], num_classes, num_classes))

    acc_list = []
    for i in range(num_classes - 1):
        for j in range(i+1, num_classes):
            logging.debug("=========== Classifying %s vs %s", i, j)
            Xa, Yb = one_vs_one_partition(X, Y, i, j)
            Xa_val, Yb_val = one_vs_one_partition(X_val, Y_val, i, j)

            classifier, acc_v, acc_list_v = c_helper(Xa, Yb,
                                                              Xa_val, Yb_val,
                                                              (i, j))
            acc_list.append(acc_list_v)

            logging.debug("")
            logging.info("=========== DONE classifying %s vs %s ============",
                         i, j)
            logging.debug("")
            cm_b = classifier.classify(Xa_val, Yb_val)
            logging.debug("%s", cm_b)
#             classifier.plot_likelihood_train(prefix="likelihood")

            # Record test prediction for later comparison
            Yp[:, i, j] = classifier.predict(X_val)
            Yp[:, j, i] = -Yp[:, i, j]

            #logger.debug("Yp: %s", Yp[:, i, j])

    logging.info("============== DONE ALL ===============")
    logging.debug("")

    Yp = np.sum(Yp, axis=2)
    logger.debug("Yp after sum: %s", Yp)
    Y_guess = np.argmax(Yp, axis=1)

    c_matrix = util.confusion_matrix(Y_val, Y_guess, num_classes)

    logging.info('Overall test acc: %f%%', util.get_accuracy(c_matrix))
    logging.debug('%s', c_matrix)
    logging.debug(".......")
    
    return c_matrix, acc_list

def spread(Y, num_classes):
    Y_row = np.array(range(Y.shape[0]))
    Y_ = np.zeros((Y.shape[0], num_classes), dtype=np.int8)
    Y_[Y_row, Y] = 1
    return Y_

def babysit_nn(X, Y, X_val, Y_val, num_classes, hidden_layer_sizes,
               learning_rate, reg_constant, num_iterations):
    Y_spread = spread(Y, num_classes)
    Y_val_spread = spread(Y_val, num_classes)

    from ml_lib.n_net import NNet
    nn = NNet(X, Y_spread, hidden_layer_sizes, 0, learning_rate, reg_constant)

    ITERS_PER_RUN = 5

    J = []      # output cost
    J_r = []    # regularization cost
    Sat_s = []  # %age saturated
    TAcc_s = [] # traing accuracy
    VAcc_s = []
    It = []
    for train_loops in range(num_iterations // ITERS_PER_RUN):
        (it, error_cost, reg_cost, sat, tr_acc) = nn.train(ITERS_PER_RUN)
        It.append(it)
        J.append(error_cost)
        J_r.append(reg_cost)
        Sat_s.append(sat)
        TAcc_s.append(tr_acc)
        
        val_acc = util.get_accuracy(
            nn.classify(X_val, Y_val_spread, should_report=False))
        VAcc_s.append(val_acc)
        
        logging.debug("Iteration %d. Cost=%0.2f \t Tr=%0.2f Val=%0.2f Sat=%0.2f",
                      it, error_cost + reg_cost, tr_acc, val_acc, sat)

    # Plot Cost function
    util.plot_all(It,
                  #[J], 
                  #[J, J_r], 
                  [J, Sat_s, TAcc_s, VAcc_s],
                  #[TAcc_s, VAcc_s],
                  "iteration", "",
                  ["Cost", "Saturation", "Training", "Validation"],
                  #["Training", "Validation"],
                  nn.prefix())
    

    
    