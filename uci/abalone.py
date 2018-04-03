'''
@author enerve

'''
from __future__ import division
import logging
import math
import numpy as np

from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier
import ml_lib.util as util
import ml_lib.data_util as data_util
import ml_lib.helper as helper
import ml_lib.cmd_line as cmd_line
import ml_lib.log as log

# Draw histograms for each column
def draw_classes_histogram(X, Y, num_classes):
    for col in range(10):
        util.draw_class_histograms(X, Y, num_classes, col)


def main():
    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_dataset = "abalone"

    logger = logging.getLogger()
    log.configure_logger(logger, util.pre_dataset)
    logger.setLevel(logging.DEBUG)
    
    logger.info("--- Abalone dataset ---")

    data = np.genfromtxt(args.file, delimiter=",",
                         converters={0: lambda x: 0.0 if x=='M' 
                                     else 1.0 if x=='F' 
                                     else 2.0 if x=='I' else -1.0}
                        )
    
    # Create separate column for each categorical 'sex' value
    X = np.append(np.zeros((data.shape[0], 2)), data[:,1:8], axis=1)    
    X[:, 0][data[:, 0] == 0.0] = 1
    X[:, 1][data[:, 0] == 1.0] = 1
    
    X = data_util.append_feature(data[:, 1] * data[:, 2] * data[:, 3], X)
    X = data_util.append_feature(data[:, 1] * data[:, 2], X)
    X = data_util.append_feature(data[:, 1] * data[:, 1], X)

    # X = util.select_features(X, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    Y = data[:, 8].astype(int)

    logger.debug("Full dataset: %s", X.shape)
    logger.debug("  Males:   %s", np.sum(X[:,0]))
    logger.debug("  Females: %s", np.sum(X[:,1]))
    logger.debug("  Infants: %s", np.sum(X[:,2]))
    
    X, Y, X_valid, Y_valid, X_test, Y_test = \
        data_util.split_into_train_test_sets(
            X, Y, args.validation_portion, args.test_portion)
    
    logger.debug("Training dataset:   %s", (X.shape, ))
    logger.debug("Validation dataset(%d): %s", args.validation_portion,
                 X_valid.shape)
    logger.debug("Testing dataset(%d): %s", args.test_portion, X_test.shape)

    if args.normalize:
        logger.info("Normalizing...")
        util.pre_norm = "n"
        X, X_valid, X_test = data_util.normalize_all(X, X_valid, X_test)

    if True: # Classification
        split_points = [-1, 8, 10]

        Ya, Ya_valid, Ya_test = data_util.bucketify(Y, Y_valid, Y_test,
                                                    split_points)
        data_util.describe_classes(Ya, Ya_valid, Ya_test)
        num_classes = len(split_points)

        if args.draw_classes_histogram:
            draw_classes_histogram(X, Ya, num_classes)

        if args.bayes:
            logger.info("Bayes classifier...")
            util.pre_alg = "bayes"
            # Gaussian plug-in classifier
            
            gpi_classifier = GaussianPlugInClassifier(X, Ya, num_classes)
                
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(gpi_classifier.classify(X_test, Ya_test)[0])
    
        if args.naive:
            logger.info("Naive bayes classifier...")
            util.pre_alg = "naive"
            # Gaussian naive bayes classifier
            
            naive_classifier = GaussianNaiveClassifier(X, Ya, num_classes)
            
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(naive_classifier.classify(X_test, Ya_test)[0])

        if args.perceptron:
            logger.info("Perceptron...")
            util.pre_alg = "perceptron"
            from ml_lib.perceptron import Perceptron
            
            helper.classify_one_vs_one([],
                X, Ya, X_test, Ya_test, num_classes,
                lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 8000, 0))
            
        if args.logistic:
            logger.info("Logistic Regression...")
            util.pre_alg = "logistic"
            from ml_lib.logistic import Logistic
            
            helper.classify_one_vs_one([],
                X, Ya, X_test, Ya_test, num_classes,
                lambda X, Y: Logistic(X, Y, step_size=0.001, max_steps=15000,
                                      reg_constant=0.01))

        if args.knn:
            logger.info("k-Nearest Neighbor...")
            util.pre_alg = "knn"
            from ml_lib.knn import KNN
            
            run_single_classifier = False
            if run_single_classifier: 
                knn_classifier = KNN(X, Ya, 10, 3)
                util.report_accuracy(knn_classifier.classify(X_test, Ya_test))
    
            for k in range(10, 30):
                logger.info("%s-NN", k+1)
                knn_classifier = KNN(X, Ya, 1+k, 3)
                util.report_accuracy(knn_classifier.classify(X_valid, Ya_valid))

        if args.svm:
            logger.info("Support Vector Machine...")
            util.pre_alg = "svm"
            from ml_lib.svm import SVM, RBFKernel
            from ml_lib.svm_sk_svc import SVMSkSVC
            #from ml_lib.svm_sk_linear import SVMSkLinear

            run_single_classifier = False
            if run_single_classifier: 
                i = 2
                Yb = np.zeros((Ya.shape[0]))
                Yb[Ya==i] = 1
                Yb_valid = np.zeros((Ya_valid.shape[0]))
                Yb_valid[Ya_valid==i] = 1

                svm_classifier = SVM(X, Yb, 12, kernel=RBFKernel(0.4))
                util.report_accuracy(svm_classifier.classify(X_valid, Yb_valid))

                svm_classifier = SVM(X, Yb, 14.4, kernel=RBFKernel(0.1))
                util.report_accuracy(svm_classifier.classify(X_valid, Yb_valid))

            run_onevsone_multiclassifier = False
            if run_onevsone_multiclassifier:
                helper.classify_one_vs_one([],
                    X, Ya, X_test, Ya_test, num_classes,
                    #lambda X, Y, lam=1: SVMSkLinear(X, Y, lam)
                    lambda X, Y, lam=14.4: SVM(X, Y, lam)#, kernel=RBFKernel(0.1))
                    #lambda X, Y, lam=1, b=0.5: SVMSkSVC(X, Y, lam, b, kernel='rbf')
                    )
            
            run_onevsall_multiclassifier = False
            if run_onevsall_multiclassifier:
                cm, acc_list = helper.classify_one_vs_all([],
                    X, Ya, X_valid, Ya_valid, num_classes,
                    lambda X, Y, lam=math.pow(1.5, 4)*10, b=0.85: SVM(X, Y, lam, 
                                                         kernel=RBFKernel(b)))
                util.report_accuracy(cm)

            run_rbf_one_vs_all_cross_validation = True
            if run_rbf_one_vs_all_cross_validation:
                for pre_svm_cv_x in ["b", "l"]:
                    if pre_svm_cv_x == "b":
                        lam_val = [math.pow(1.5, p+1)*10 for p in range(7)]
                        b_val = [(p+1)/20 for p in range(27)]
                    elif pre_svm_cv_x == "l":
                        lam_val = [math.pow(1.2, p+1)*10 for p in range(27)]
                        b_val = [(p+1)/10 for p in range(7)]

                    logger.debug("lambda values: %s", lam_val)
                    logger.debug("b values: %s", b_val)

                    lmbd_sk = lambda X, Y, b, lam: SVMSkSVC(X, Y, lam, b,
                                                            kernel='rbf')
                    # Use a single instance so K matrix can be shared better
                    single_svm = SVM(X)
                    lmbd_my = lambda X, Y, b, lam, svm=single_svm: \
                        svm.initialize(Y, lam, RBFKernel(b))

                    for pre_svm_alg in ["sk", "my"]:

                        cm, acc_list = helper.classify_one_vs_all(
                            [b_val, lam_val],
                            X, Ya, X_valid, Ya_valid, num_classes,
                            lmbd_sk if pre_svm_alg=="sk" else lmbd_my)
            
                        for i in range(len(acc_list)):
                            logger.info("--- Class #%d", i)
                            acc_matrix = np.array(acc_list[i])
                            #acc_matrix = np.genfromtxt(util.prefix() +
                            #                           '995985.0' + "_cv_%d.csv" % (i),
                            #                           delimiter=",")
                            logger.info("%s", acc_matrix)
                            
                            suff = "val_%s_class%d_%s"%(pre_svm_alg, i,
                                                        pre_svm_cv_x)
                            csv_filename = util.prefix() + suff + ".csv"
                            logger.debug("Writing to file %s", csv_filename)
                            np.savetxt(csv_filename,
                                       acc_matrix, delimiter=",", fmt='%.3f')
                            if pre_svm_cv_x == 'b':
                                util.plot_accuracies(acc_matrix.T, b_val,
                                                     "RBF width b", lam_val,
                                                     "Lambda (C)", suff)
                            elif pre_svm_cv_x == 'l':
                                util.plot_accuracies(acc_matrix, lam_val, 
                                                     "Lambda (C)", b_val,
                                                     "RBF width b", suff)
    
            run_rbf_multiclass = False
            if run_rbf_multiclass:
                param = {
                    0: {'b': 0.25, 'lambda': 113},
                    1: {'b': 0.6, 'lambda': 113},
                    2: {'b': 0.7, 'lambda': 75}}
                
                cm, acc_list = helper.one_vs_all_multiclassify(
                    X, Ya, X_test, Ya_test, num_classes,
                    lambda X, Y, X_val, Y_val, info:
                        helper.classifier_helper(
                            X, Y, X_val, Y_val,
                            lambda X, Y: SVM(X, Y, param[info[0]]['lambda'],
                                             kernel=RBFKernel(
                                                 param[info[0]]['b']))))
                util.report_accuracy(cm)

            run_linear_one_vs_one_cross_validation = False
            if run_linear_one_vs_one_cross_validation:
                lam_val = [math.pow(1.2, p+1)*10 for p in range(27)]
                logger.debug("lambda values: %s", lam_val)

                cm, acc_list = helper.classify_one_vs_one([lam_val],
                    X, Ya, X_valid, Ya_valid, num_classes, SVM)
    
                #TODO: move inside helper.onevsone
                ai = 0
                for i in range(num_classes - 1):
                    for j in range(i+1, num_classes):
                        logger.info("--- Class %d vs %d", i, j)
                        acc_matrix = np.array(acc_list[ai])
                        ai += 1
                        logger.info("%s", acc_matrix)
                        suff = "val_class%dx%d"%(i, j)
                        csv_filename = util.prefix() + suff + ".csv"
                        logger.debug("Writing to file %s", csv_filename)
                        np.savetxt(csv_filename,
                                   acc_matrix, delimiter=",", fmt='%.3f')
                ai = 0
                for i in range(num_classes - 1):
                    for j in range(i+1, num_classes):
                        acc_matrix = np.array(acc_list[ai])
                        ai += 1
                        suff = "val_class%dx%d"%(i, j)
                        util.plot_accuracy(acc_matrix, lam_val, None, suff)

            run_linear_one_vs_one = False
            if run_linear_one_vs_one:
                lam_val = [[0, 25, 550],
                           [0,  0, 600],
                           [0,  0,   0]]
                
                cm, acc_list = helper.one_vs_one_multiclassify(
                    X, Ya, X_test, Ya_test, num_classes,
                    lambda X, Y, X_val, Y_val, info:
                        helper.classifier_helper(
                            X, Y, X_val, Y_val,
                            lambda X, Y: SVM(X, Y, lam_val[info[0]][info[1]])))
                util.report_accuracy(cm)

            run_rbf_one_vs_one_cross_validation = False
            if run_rbf_one_vs_one_cross_validation:
                for pre_svm_cv_x in ["l", "b"]:
                    if pre_svm_cv_x == "b":
                        lam_val = [math.pow(2, p+1)*10 for p in range(7)]
                        b_val = [(p+1)/10 for p in range(27)]
                    elif pre_svm_cv_x == "l":
                        lam_val = [math.pow(1.2, p+1)*10 for p in range(27)]
                        b_val = [(p+1)/2.5 for p in range(7)]

                    logger.debug(lam_val)
                    logger.debug(b_val)

                    # Use a single instance so K matrix can be shared better
    #                     single_svm = SVM(X)
                    lmbd_my = lambda X, Y, b, lam: \
                        SVM(X).initialize(Y, lam, RBFKernel(b))
                     
                    cm, acc_list = helper.classify_one_vs_one([b_val, lam_val],
                        X, Ya, X_valid, Ya_valid, num_classes, lmbd_my)
         
                    #TODO: move inside helper.onevsone
                    ai = 0
                    for i in range(num_classes - 1):
                        for j in range(i+1, num_classes):
                            suff = "val_class%dx%d_%s"%(i, j, pre_svm_cv_x)
                            logger.info("--- Class %d vs %d", i, j)
                            acc_matrix = np.array(acc_list[ai])
                            logger.info("%s", acc_matrix)
                            np.savetxt(util.prefix() + suff + ".csv",
                                       acc_matrix, delimiter=",", fmt='%.3f')
                            ai += 1
    
                    ai = 0
                    for i in range(num_classes - 1):
                        for j in range(i+1, num_classes):
                            suff = "val_class%dx%d_%s"%(i, j, pre_svm_cv_x)
                            acc_matrix = np.array(acc_list[ai])
    #                         acc_matrix = np.genfromtxt(util.prefix('916380') +
    #                                                    suff + ".csv",
    #                                                    delimiter=",")
                            if pre_svm_cv_x == 'b':
                                util.plot_accuracies(acc_matrix.T, b_val,
                                                     "RBF width b", lam_val,
                                                     "Lambda (C)", suff)
                            elif pre_svm_cv_x == 'l':
                                util.plot_accuracies(acc_matrix, lam_val, 
                                                     "Lambda (C)", b_val,
                                                     "RBF width b", suff)
                            ai += 1

            run_rbf_one_vs_one = False
            if run_rbf_one_vs_one:
                b_val = [[0, 0.7,  1],
                         [0,   0,  2], 
                         [0,   0,  0]]
                lam_val = [[0, 30, 1400],
                           [0,  0,  320],
                           [0,  0,    0]]
                
                cm, acc_list = helper.one_vs_one_multiclassify(
                    X, Ya, X_test, Ya_test, num_classes,
                    lambda X, Y, X_o, Y_o, info:
                        helper.classifier_helper(
                            X, Y, X_o, Y_o,
                            lambda X, Y: SVM(X, Y, lam_val[info[0]][info[1]],
                                             RBFKernel(b_val[info[0]][info[1]]))
                            ))
                util.report_accuracy(cm)

if __name__ == '__main__':
    main()