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
        self.alphas = None
        self.intercept = None

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
    def train(self):
        """
        train svm and get the hyper plane parameters
        """
        alphas, intercept = Svm.smo_simple(self.train_features, self.train_labels, 0.6, 0.001, 40)
        self.alphas = alphas
        self.intercept = intercept
        print alphas
        print intercept
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
        features_mat = np.asmatrix(features).astype(float)
        labels_mat = np.asmatrix(labels).astype(float).transpose()
        # intercept of hyperplane function
        inter = 0.0
        sample_num, feature_num = np.shape(features_mat)
        # init alphas
        alphas = np.mat(np.zeros((sample_num, 1)).astype(float))
        cur_iter = 0
        while cur_iter < iters:
            alpha_pairs_changed = 0
            for i in range(sample_num):
                f_x_i = np.multiply(alphas, labels_mat).T * \
                    (features_mat * features_mat[i, :].T) + inter
                e_i = f_x_i - labels_mat[i]
                if ((labels_mat[i] * e_i < -tolerance) and (alphas[i] < uper)) or \
                    ((labels_mat[i] * e_i > tolerance) and (alphas[i] > 0)):
                    j = Svm.select_j_rand(i, feature_num)
                    f_x_j = np.multiply(alphas, labels_mat).T * \
                        (features_mat * features_mat[j, :].T) + inter
                    e_j = f_x_j - labels_mat[j]
                    alpha_i_old = alphas[i].copy()
                    alpha_j_old = alphas[j].copy()
                    if labels_mat[i] != labels_mat[j]:
                        low = max(0.0, alphas[j] - alphas[i])
                        high = min(uper, uper + alphas[j] - alphas[i])
                    else:
                        low = max(0.0, alphas[j] + alphas[i] - uper)
                        high = min(uper, alphas[j] + alphas[i])
                    if low == high:
                        print "low == high"
                        continue
                    eta = 2.0 * features_mat[i, :] * features_mat[j, :].T - \
                        features_mat[i, :] * features_mat[i, :].T - \
                        features_mat[j, :] * features_mat[j, :].T
                    if eta >= 0:
                        print "eta >= 0"
                        continue
                    alphas[j] -= labels_mat[j] * (e_i - e_j) / eta
                    alphas[j] = Svm.clip_alpha(alphas[j], high, low)

                    if abs(alphas[j] - alpha_j_old) < 0.00001:
                        print "j not moving enough"
                        continue

                    alphas[i] += labels_mat[j] * labels_mat[i] * (alpha_j_old - alphas[j])

                    b_1 = inter - e_i - labels_mat[i] * (alphas[i] - alpha_i_old) * \
                        features_mat[i, :] * features_mat[i, :].T - \
                        labels_mat[j] * (alphas[j] - alpha_j_old) * \
                        features_mat[i, :] * features_mat[j, :].T

                    b_2 = inter - e_j - labels_mat[i] * (alphas[i] - alpha_i_old) * \
                        features_mat[i, :] * features_mat[i, :].T - \
                        labels_mat[j] * (alphas[j] - alpha_j_old) * \
                        features_mat[j, :] * features_mat[j, :].T

                    if alphas[i] > 0 and alphas[i] < uper:
                        inter = b_1
                    elif alphas[j] > 0 and alphas[j] < uper:
                        inter = b_2
                    else:
                        inter = (b_1 + b_2) / 2.0
                    alpha_pairs_changed += 1
                    print "current iteration: %d i:%d, pairs changed %d" % \
                        (cur_iter, i, alpha_pairs_changed)
            if alpha_pairs_changed == 0:
                cur_iter += 1
            else:
                cur_iter = 0
            print "iteration number: %d" % cur_iter
        return alphas, inter


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
