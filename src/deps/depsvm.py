# conding=utf-8
"""
deps func like plot for svm
"""
import numpy as np

class smo_struct(object):
    """
    smo optimize struct
    """
    def __init__(self, features, labels, uper, tolerance):
        self.features = features
        self.lables = labels
        self.uper = uper
        self.tolerance = tolerance
        self.samples = np.shape(features)[0]
        self.alphas = np.mat(np.zeros((self.samples, 1)))
        self.intercept = 0.0
        self.error_cache = np.mat(np.zeros((self.samples, 2)))

def cal_error_k(ss, k):
    """
    calculate alpha k error 
    :param ss: smo_struct
        smo optimize struct 
    :param k: int
        int number k
    """
    # f_x_k = np.multipy(ss.alphas, ss.labels).T * (ss.features * ss.features[k, :].T) + ss.intercept
    return None
