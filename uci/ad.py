'''
Created on Apr 3, 2018

@author enerve
'''
from __future__ import division
import logging
import math
import numpy as np
import pandas as pd

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
    util.pre_dataset = "ad"

    logger = logging.getLogger()
    log.configure_logger(logger, util.pre_dataset)
    logger.setLevel(logging.DEBUG)
    
    logger.info("--- Internet Ads dataset ---")

    # Number of Attributes: 1558 (3 continous; others binary
    # One or more of the three continous features are missing in 28%
    # of the instances
    # output: "ad" or "nonad"
    
#     data = np.genfromtxt(args.file, delimiter=",",
#                          converters={
#                              0: lambda x: float(x) if str(x).endswith('?') else -1.0,
#                              1: lambda x: float(x) if str(x).endswith('?') else -1.0,
#                              2: lambda x: float(x) if str(x).endswith('?') else -1.0,
#                              1558: lambda x: 1.0 if str(x)=="ad." else -1.0
#                          },
#                          #autostrip=True
#                         )
#      
#     logger.debug("%s", data[0])
#     logger.debug("%s", data.shape)
    
    df = pd.read_csv(args.file, 
                     dtype={0: str, 1: str, 2: str, 3: str, 1558: str},
                     header=None)
    
    df = df.replace('?', np.nan)
    df = df.replace('   ?', np.nan)
    df = df.replace('     ?', np.nan)
    df = df.replace('nonad.', 0)
    df = df.replace('ad.', 1)
#     df = df.replace({
#             0: {'   ?': np.nan},
#             1: {'   ?': np.nan},
#             2: {'     ?': np.nan},
#             3: {'?': np.nan},
#             1558: {'ad.': 1,
#                    'nonad.': 0},
#         })
    
#     logger.debug("%s" % df[df == '?'])
    df[[0, 1, 2, 3, 1558]] = df[[0, 1, 2, 3, 1558]].apply(pd.to_numeric)

#     logger.debug("%s", df.mean())
#     logger.debug("%s", df.groupby(0).mean())

    logger.debug("Full dataset: %s", df.shape)
    logger.debug("      #Ads: %s", np.sum(df.loc[:, 1558] == 1.0))
    logger.debug("  #Non-ads: %s", np.sum(df.loc[:, 1558] == 0.0))
    
    logger.debug("=======")

    # Split dataframe in training/val/test sets
    
    SEED = 10
    logger.debug("Random seed: %d" % SEED)
    np.random.seed(SEED)
    msk_rand = np.random.rand(len(df))
    df_train = df[msk_rand <= 0.5]
    
    ad_mean = df_train.loc[df.loc[:, 1558] == 1.0].mean()
    nonad_mean = df_train.loc[df.loc[:, 1558] == 0.0].mean()

    # Fill missing values with class-specific means
    # TODO: remove?
    df_1 = df.loc[df.loc[:, 1558] == 1.0].fillna(ad_mean)
    df_0 = df.loc[df.loc[:, 1558] == 0.0].fillna(nonad_mean)
    df_c = pd.concat([df_0, df_1])
    
    df_c = df_c.sample(frac=1.0, random_state=20) # shuffle rows

    df_train = df_c[msk_rand <= 0.5]
    df_valid = df_c[(msk_rand > 0.5) & (msk_rand <= 0.75)]
    df_test = df_c[msk_rand > 0.75]
    
#     logger.debug("Valid:\n%s", df_valid.tail())
#     logger.debug("Test:\n%s", df_test.tail())
    
#     # drop duplicate rows
#     df = df.drop_duplicates()
#     # drop duplicate columns (TODO: ideally keep track of which cols dropped)
#     df = df.T.drop_duplicates().T
    
#     # randomize
#     df2 = df_c.sample(frac=1, random_state=1)
# 
#     logger.debug("\n%s", df2.describe())
#     
#     logger.debug("%s", df2.loc[:,1558].sum())
#     
#     data = df2.values

#     X = data[:, 0:-1]
#     
#     l1 = 460
#     logger.debug("l1 %s", np.sum(X[:, 3:l1], dtype=int, axis=1))
#     l2 = l1 + 495
#     logger.debug("l2 %s", np.sum(X[:, l1:l2], dtype=int, axis=1))
#     l3 = l2 + 472
#     logger.debug("l3 %s", np.sum(X[:, l2:l3], dtype=int, axis=1))
#     l4 = l3 + 111
#     logger.debug("l4 %s", np.sum(X[:, l3:l4], dtype=int, axis=1))
#     logger.debug("l5 %s", np.sum(X[:, l4:], dtype=int, axis=1))
    
#     # Drop columns that have a fixed value
#     # (TODO: ideally keep track of which cols dropped)
#     X_range = np.max(X, axis=0) - np.min(X, axis=0)
#     logger.debug("%s", X.shape)
#     X = X[:, X_range > 0]
#     logger.debug("%s", X.shape)
#     #logger.debug("%s", np.sum(X, axis=0, dtype=int))

#     Y = data[:, -1].astype(int)

#     for end in range(X.shape[0]):
#         s = np.sum(X[:end], axis=0)
#         if np.count_nonzero(s) == 0:
#             X_train = X[:end]
    
#     X, Y, X_valid, Y_valid, X_test, Y_test = \
#         data_util.split_into_train_test_sets(
#             X, Y, args.validation_portion, args.test_portion)

    X = df_train.values[:, 0:-1]
    Y = df_train.values[:, -1].astype(int)
    logger.debug("==>%s", Y.dtype)
    X_valid = df_valid.values[:, 0:-1]
    Y_valid = df_valid.values[:, -1].astype(int)
    X_test = df_test.values[:, 0:-1]
    Y_test = df_test.values[:, -1].astype(int)
    
    logger.debug("Training dataset:   %s", (X.shape, ))
    logger.debug("Validation dataset(%d): %s", args.validation_portion,
                 X_valid.shape)
    logger.debug("Testing dataset(%d): %s", args.test_portion, X_test.shape)

    if args.normalize:
        logger.info("Normalizing...")
        util.pre_norm = "n"
        X, X_valid, X_test = data_util.normalize_all(X, X_valid, X_test)

    if True: # Classification

        num_classes = 2
        data_util.describe_classes(num_classes, Y, Y_valid, Y_test)

        if args.draw_classes_histogram:
            draw_classes_histogram(X, Y, num_classes)

        if args.bayes:
            logger.info("Bayes classifier...")
            util.pre_alg = "bayes"
            from ml_lib.gaussian_plugin_classifier import GaussianPlugInClassifier 
            
            gpi_classifier = GaussianPlugInClassifier(X, Y, num_classes)
                
            # util.report_accuracy(gpi_classifier.classify(X, Y, 0.5)[0])
            util.report_accuracy(gpi_classifier.classify(X_test, Y_test)[0])
    
        if args.naive:
            logger.info("Naive bayes classifier...")
            util.pre_alg = "naive"
            from ml_lib.gaussian_naive_classifier import GaussianNaiveClassifier
            
            naive_classifier = GaussianNaiveClassifier(X, Y, num_classes)
            
            util.report_accuracy(naive_classifier.classify(X, Y)[0])
#             util.report_accuracy(naive_classifier.classify(X_test, Y_test)[0])

        if args.perceptron:
            logger.info("Perceptron...")
            util.pre_alg = "perceptron"
            from ml_lib.perceptron import Perceptron
            
            helper.classify_one_vs_one([],
                X, Y, X_test, Y_test, num_classes,
                lambda X, Y: Perceptron(X, Y, args.stochastic, 1, 8000, 0))
            
        if args.logistic:
            logger.info("Logistic Regression...")
            util.pre_alg = "logistic"
            from ml_lib.logistic import Logistic
            
            
            regs = [math.pow(1.2, p) * 10 for p in range(10)]

            helper.classify_one_vs_one([regs],
                X, Y, X_test, Y_test, num_classes,
                lambda X, Y, reg: Logistic(X, Y, step_size=0.01, max_steps=300,
                                      reg_constant=reg))

        if args.knn:
            logger.info("k-Nearest Neighbor...")
            util.pre_alg = "knn"
            from ml_lib.knn import KNN
            
            run_single_classifier = False
            if run_single_classifier: 
                knn_classifier = KNN(X, Y, 10, 3)
                util.report_accuracy(knn_classifier.classify(X_test, Y_test))
    
            for k in range(10, 30):
                logger.info("%s-NN", k+1)
                knn_classifier = KNN(X, Y, 1+k, 3)
                util.report_accuracy(knn_classifier.classify(X_valid, Y_valid))

        if args.svm:
            logger.info("Support Vector Machine...")
            util.pre_alg = "svm"
            from ml_lib.svm import SVM, RBFKernel
            from ml_lib.svm_sk_svc import SVMSkSVC
            #from ml_lib.svm_sk_linear import SVMSkLinear

            run_single_classifier = False
            if run_single_classifier: 
                i = 2
                Yb = np.zeros((Y.shape[0]))
                Yb[Y==i] = 1
                Yb_valid = np.zeros((Y_valid.shape[0]))
                Yb_valid[Y_valid==i] = 1

                svm_classifier = SVM(X, Yb, 12, kernel=RBFKernel(0.4))
                util.report_accuracy(svm_classifier.classify(X_valid, Yb_valid))

                svm_classifier = SVM(X, Yb, 14.4, kernel=RBFKernel(0.1))
                util.report_accuracy(svm_classifier.classify(X_valid, Yb_valid))

            run_onevsone_multiclassifier = False
            if run_onevsone_multiclassifier:
                helper.classify_one_vs_one([],
                    X, Y, X_test, Y_test, num_classes,
                    #lambda X, Y, lam=1: SVMSkLinear(X, Y, lam)
                    lambda X, Y, lam=14.4: SVM(X, Y, lam)#, kernel=RBFKernel(0.1))
                    #lambda X, Y, lam=1, b=0.5: SVMSkSVC(X, Y, lam, b, kernel='rbf')
                    )
            
            run_onevsall_multiclassifier = False
            if run_onevsall_multiclassifier:
                cm, acc_list = helper.classify_one_vs_all([],
                    X, Y, X_valid, Y_valid, num_classes,
                    lambda X, Y, lam=math.pow(1.5, 4)*10, b=0.85: SVM(X, Y, lam, 
                                                         kernel=RBFKernel(b)))
                util.report_accuracy(cm)

            run_rbf_one_vs_all_cross_validation = False
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
                            X, Y, X_valid, Y_valid, num_classes,
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
                    X, Y, X_test, Y_test, num_classes,
                    lambda X, Y, X_val, Y_val, info:
                        helper.classifier_helper(
                            X, Y, X_val, Y_val,
                            lambda X, Y: SVM(X, Y, param[info[0]]['lambda'],
                                             kernel=RBFKernel(
                                                 param[info[0]]['b']))))
                util.report_accuracy(cm)

            run_linear_one_vs_one_cross_validation = False
            if run_linear_one_vs_one_cross_validation:
                lam_val = [math.pow(1.2, p) / 5 for p in range(20)]
                logger.debug("lambda values: %s", lam_val)

                cm, acc_list = helper.classify_one_vs_one([lam_val],
                    X, Y, X_valid, Y_valid, num_classes, SVM)
    
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
                logger.debug("\n-------------------- ")
                cm, acc_list = helper.one_vs_one_multiclassify(
                    X, Y, X_test, Y_test, num_classes,
                    lambda X, Y, X_o, Y_o, info:
                        helper.classifier_helper(
                            X, Y, X_o, Y_o,
                            lambda X, Y: SVM(X, Y, 1.486)))
                util.report_accuracy(cm)

            run_rbf_one_vs_one_cross_validation = True
            if run_rbf_one_vs_one_cross_validation:
                for pre_svm_cv_x in ["l", "b"]:
                    if pre_svm_cv_x == "b":
                        lam_val = [math.pow(2, p+1) for p in range(3)]
                        b_val = [(p+10)/5 for p in range(10)]
                    elif pre_svm_cv_x == "l":
                        lam_val = [math.pow(1.2, p+1) for p in range(10)]
                        b_val = [(p+5)/2.5 for p in range(3)]

                    logger.debug(lam_val)
                    logger.debug(b_val)

                    # Use a single instance so K matrix can be shared better
    #                     single_svm = SVM(X)
                    lmbd_my = lambda X, Y, b, lam: \
                        SVM(X).initialize(Y, lam, RBFKernel(b))
                    lmbd_sk = lambda X, Y, b, lam: \
                        SVMSkSVC(X, Y, lam, b, kernel='rbf')
                     
                    cm, acc_list = helper.classify_one_vs_one([b_val, lam_val],
                        X, Y, X_valid, Y_valid, num_classes, lmbd_my)
         
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
                    X, Y, X_test, Y_test, num_classes,
                    lambda X, Y, X_o, Y_o, info:
                        helper.classifier_helper(
                            X, Y, X_o, Y_o,
                            lambda X, Y: SVM(X, Y, lam_val[info[0]][info[1]],
                                             RBFKernel(b_val[info[0]][info[1]]))
                            ))
                util.report_accuracy(cm)

        if args.nn:
            logger.info("Neural network...")
            util.pre_alg = "neural"
            #from ml_lib.n_net import NNet, NNetClassifier
            
            #regs = [math.pow(1.2, p) * 10 for p in range(10)]
            
            #logger.debug("2==>%s", Y.dtype)
            #TODO: NNs should not need this ovo-classification scheme
#             helper.classify_one_vs_one([],
#                 X, Y, X_test, Y_test, num_classes,
#                 lambda X, Y: NNetClassifier(X, Y, [100], max_steps=200,
#                                 learning_rate=10, reg_constant=0.1,
#                                 num_classes=2))
            
            helper.babysit_nn(X, Y, X_test, Y_test, num_classes,
                              hidden_layer_sizes=[100],
                              learning_rate=10, reg_constant=0.1,
                              num_iterations=400)
            
            

if __name__ == '__main__':
    main()