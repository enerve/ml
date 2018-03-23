'''
Created on Mar 20, 2018

@author: enerve
'''

import logging
import sys
import time

import ml_lib
import ml_lib.util as util

def configure_logger(logger, name):
    # create file handler which logs even debug messages
    log_filename = (util.pre_outputdir if util.pre_outputdir else '') + \
                    'log/output' + \
                    '_%s' % name + \
                    '_%s' % str(int(round(time.time()) % 10000000)) + \
                    '.log'
    print "Logging to %s" % log_filename
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '(%(name)s) %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    ml_lib.data_util.init_logger()
    ml_lib.helper.init_logger()
    ml_lib.util.init_logger()
