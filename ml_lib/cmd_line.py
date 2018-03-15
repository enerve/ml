'''
Created on Mar 15, 2018

@author: erw
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('file', help='path to data file')
    parser.add_argument('--output_dir', help='path to store output files')
    parser.add_argument('--test_portion',
                        help='Which portion to use as test set',
                        default=1, type=int)
    parser.add_argument('--validation_portion',
                        help='Which portion to use as validation set',
                        default=2, type=int)
    parser.add_argument('--draw_classes_data', action='store_true')
    parser.add_argument('--draw_classes_histogram', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--bayes', action='store_true')
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--perceptron', action='store_true')
    parser.add_argument('--sklearn_perceptron', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--logistic', action='store_true')
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--svm', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    pass