# coding=utf-8
"""
Author: shengyitao
Create Date: 2017-01-05
"""

import numpy as np

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
        return None