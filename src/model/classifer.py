# coding=utf-8
"""
Author: shengyitao
Create Date: 2017-01-05
"""

import numpy as np
import random

methods = ['svm', 'k-means']

class Svm(object):
    """
    None
    """
    def __init__(self, k='linear', alg='smo'):
        self.kernel = k
        self.alg = alg
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

    def load_data(self, feats, labels, spec='train'):
        """
        load data, include feature values and labels into self
        :param spec:
            'train': load training data
            'test': load test data
        :param feats:
            [
                [v11, v12, v13, ......, v1n],
                [v21, v22, v23, ......, v2n],
                .
                .
                [vm1, vm2, vm3, ......, vmn]
            ]
            'm' is the number of rows(data number)
            'n' is the number of columns(feature number)
        :param labels:
            [1, 1, -1, ......, 1]
            length = m
        :return:None
        """
        if spec == 'train':
            self.train_features = feats
            self.train_labels = labels
        elif spec == 'test':
            self.test_features = feats
            self.test_labels = labels

    @staticmethod
    def smo(features, labels):
        """
        :param features:
            [
                [v11, v12, v13, ......, v1n],
                [v21, v22, v23, ......, v2n],
                .
                .
                [vm1, vm2, vm3, ......, vmn]
            ]
        :param labels:
            [1, 1, -1, ......, 1]
            length = m
        """
        if features is None or labels is None:
            print "no input data"
            return None
        return None
    @staticmethod
    def smo_simple(features, labels, uper, tolerance, iters):
        """
        :param features:
            [
                [v11, v12, v13, ......, v1n],
                [v21, v22, v23, ......, v2n],
                .
                .
                [vm1, vm2, vm3, ......, vmn]
            ]
        :param labels:
            [1, 1, -1, ......, 1]
            length = m
        :param uper:
            0<alpha<uper
        :param tolerance:
        :param iters:
            max iterate times
        """
        features_mat = np.mat(features)
        labels_mat = np.mat(labels).transpose()
        b = 0
        samples, feature_num = np.shape(features_mat)
        # init alphas
        alphas = np.mat(np.zeros((samples, 1)))
        cur_iter = 0
        while cur_iter < iters:
            alpha_pairs_changed = 0

    @staticmethod
    def select_j_rand(alpha_i, sample_num):
        """
        select j randomly from 0 to sample_num and diff to alpha_i
        :params alpha_i: int
            int number
        :param sample_num: int
            number of sample(not small than alpha_i)
        :returns:
            alpha_j: int
                int number
        """
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, sample_num))

        return alpha_j

    @staticmethod
    def clip_alpha(alpha_j, high, low):
        """
        gurantee alpha_j in range [low, high]
        :param alpha_j: int
            int number
        :param high: int
            int number(uper limit)
        :param low: int
            int number(lower limit)
        :returns:
            alpha_j
        """
        if alpha_j > high:
            alpha_j = high

        if alpha_j < low:
            alpha_j = low

        return alpha_j