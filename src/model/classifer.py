# coding=utf-8
"""
Author: shengyitao
Create Date: 2017-01-05
"""

import numpy as np
import random
methods = ['svm', 'k-means']

class Svm_simple(object):
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
        self.weights = None

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
            'n' is the number of columns(f eature number)
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
        if self.train_features is None or self.train_labels is None:
            print "no training data"
            return None
        alphas, intercept = Svm.smo_simple(self.train_features, self.train_labels, 1.0, 0.001, 30)
        self.alphas = alphas
        self.intercept = intercept
        self.weights = self.get_w()

    def get_w(self):
        """
        get weight of hyper plane
        """
        features = np.asmatrix(self.train_features).astype(float)
        labels = np.asmatrix(self.train_labels).astype(float)
        alphas = self.alphas
        samples, feature_num = np.shape(features)
        weights = np.zeros((feature_num, 1))
        for i in range(samples):
            weights += np.multiply(alphas[i][0] * labels[0, i], features[i, :].T)
        return weights

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
        :param features:numpy array
            [
                [v11, v12, v13, ......, v1n],
                [v21, v22, v23, ......, v2n],
                .
                .
                [vm1, vm2, vm3, ......, vmn]
            ]
        :param labels:numpy array
            [1, 1, -1, ......, 1]
            length = m
        :param uper:float
            0<=alpha<=uper(find alpha s.t. 0<alpha<C), is called penalty parameter
        :param tolerance:float
            the error can be tolerated
        :param iters:int
            max iterate times
        """
        features_mat = np.asmatrix(features).astype(float)
        labels_mat = np.asmatrix(labels).astype(float).transpose()
        # intercept of hyperplane function
        inter = 0.0
        sample_num, feature_num = np.shape(features_mat)
        del feature_num
        # init alphas
        alphas = np.mat(np.zeros((sample_num, 1)).astype(float))
        cur_iter = 0
        while cur_iter < iters:
            alpha_pairs_changed = 0
            print "******"
            print cur_iter
            print "******"
            for i in range(sample_num):
                # print "*******"
                # print i
                # print "*******"
                f_x_i = np.multiply(alphas, labels_mat).T * \
                    (features_mat * features_mat[i, :].T) + inter
                e_i = f_x_i - labels_mat[i]
                if ((labels_mat[i] * e_i < -tolerance) and (alphas[i] < uper)) or \
                    ((labels_mat[i] * e_i > tolerance) and (alphas[i] > 0)):
                    j = Svm.select_j_rand(i, sample_num)
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
                        # print "low == high"
                        continue
                    eta = 2.0 * features_mat[i, :] * features_mat[j, :].T - \
                        features_mat[i, :] * features_mat[i, :].T - \
                        features_mat[j, :] * features_mat[j, :].T
                    if eta >= 0:
                        # print "eta >= 0"
                        continue
                    alphas[j] -= labels_mat[j] * (e_i - e_j) / eta
                    alphas[j] = Svm.clip_alpha(alphas[j], high, low)

                    if abs(alphas[j] - alpha_j_old) < 0.00001:
                        # print "j not moving enough"
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
            # cur_iter += 1
            # print "iteration number: %d" % cur_iter
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
class SvmComplete(object):
    """
    full svm
    """
    def __init__(self, features, labels, uper, tolerance, max_iter=40):
        self.features_mat = np.asmatrix(features).astype(float)
        self.labels_mat = np.asmatrix(labels).astype(float)
        self.uper = uper
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.intercept = None
        self.alphas = None
        self.weights = None
    def fit(self):
        self.ss = self.smo_struct(self.features_mat, self.labels_mat.transpose(), self.uper, self.tolerance)
        self.intercept, self.alphas = SvmComplete.smo(self.ss, self.max_iter)
        self.weights = SvmComplete.get_w(self.features_mat, self.labels_mat, self.alphas)
    class smo_struct(object):
        """
        smo optimize struct
        """
        def __init__(self, features, labels, uper, tolerance):
            self.features = features
            self.labels = labels
            self.uper = uper
            self.tolerance = tolerance
            self.samples = np.shape(features)[0]
            self.alphas = np.mat(np.zeros((self.samples, 1)))
            self.intercept = 0.0
            self.error_cache = np.mat(np.zeros((self.samples, 2)))
    @staticmethod
    def cal_error_k(ss, k):
        """
        calculate alpha k error
        :param ss: smo_struct
            smo optimize struct
        :param k: int
            int number k
        """
        f_x_k = np.multiply(ss.alphas, ss.labels).T * (ss.features * ss.features[k, :].T) + ss.intercept
        e_k = f_x_k - ss.labels[k]
        return e_k
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
    def select_j(i, ss, e_i):
        max_k = -1
        max_delta_e = 0.0
        e_j = 0.0
        ss.error_cache[i] = [1, e_i]
        valid_e_cache_list = np.nonzero(ss.error_cache[:, 0].A)[0]
        if len(valid_e_cache_list) > 1:
            for k in valid_e_cache_list:
                if k == i:
                    continue
                e_k = SvmComplete.cal_error_k(ss, k)
                delta_e = abs(e_i - e_k)
                if delta_e > max_delta_e:
                    max_k = k
                    max_delta_e = delta_e
                    e_j = e_k
            return max_k, e_j
        else:
            j = SvmComplete.select_j_rand(i, ss.samples)
            e_j = SvmComplete.cal_error_k(ss, j)

        return j, e_j
    @staticmethod
    def update_e_k(ss, k):
        """
        update error of k
        """
        e_k = SvmComplete.cal_error_k(ss, k)
        ss.error_cache[k] = [1, e_k]
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
    @staticmethod
    def inner_loop(i, ss):
        e_i = SvmComplete.cal_error_k(ss, i)
        if (ss.labels[i] * e_i < -ss.tolerance and ss.alphas[i] < ss.uper) or \
            (ss.labels[i] * e_i > ss.tolerance and ss.alphas[i] > 0):
            j, e_j = SvmComplete.select_j(i, ss, e_i)
            alpha_i_old = ss.alphas[i].copy()
            alpha_j_old = ss.alphas[j].copy()
            if ss.labels[i] != ss.labels[j]:
                low = max(0.0, ss.alphas[j] - ss.alphas[i])
                high = min(ss.uper, ss.uper + ss.alphas[j] - ss.alphas[i])
            else:
                low = max(0.0, ss.alphas[j] + ss.alphas[i] - ss.uper)
                high = min(ss.uper, ss.alphas[j] + ss.alphas[i])

            if low == high:
                print "low == high"
                return 0
            eta = 2.0 * ss.features[i, :] * ss.features[j, :].T - \
                ss.features[i, :] * ss.features[i, :].T - \
                ss.features[j, :] * ss.features[j, :].T
            if eta >= 0:
                print "eta >= 0"
                return 0
            ss.alphas[j] -= ss.labels[j] * (e_i - e_j) / eta
            ss.alphas[j] = SvmComplete.clip_alpha(ss.alphas[j], high, low)
            SvmComplete.update_e_k(ss, j)
            if abs(ss.alphas[j] - alpha_j_old) < 0.00001:
                print "j not moving enough"
                return 0
            ss.alphas[i] += ss.labels[j] * ss.labels[i] * (alpha_j_old - ss.alphas[j])
            SvmComplete.update_e_k(ss, i)
            b_1 = ss.intercept - e_i - ss.labels[i] * (ss.alphas[i] - alpha_i_old) * \
                ss.features[i, :] * ss.features[i, :].T - \
                ss.labels[j] * (ss.alphas[j] - alpha_j_old) * \
                ss.features[i, :] * ss.features[j, :].T

            b_2 = ss.intercept - e_j - ss.labels[i] * (ss.alphas[i] - alpha_i_old) * \
                ss.features[i, :] * ss.features[i, :].T - \
                ss.labels[j] * (ss.alphas[j] - alpha_j_old) * \
                ss.features[j, :] * ss.features[j, :].T

            if ss.alphas[i] > 0 and ss.alphas[i] < ss.uper:
                ss.intercept = b_1
            elif ss.alphas[j] > 0 and ss.alphas[j] < ss.uper:
                ss.intercept = b_2
            else:
                inter = (b_1 + b_2) / 2.0
            return 1
        else:
            return 0
    @staticmethod
    def smo(ss, max_iter, k_tube=('lin', 0)):
        # ss = smo_struct(features, labels.transpose(), uper, tolerance)
        cur_iter = 0
        entire_set = True
        alpha_paired_changed = 0
        while cur_iter < max_iter and (alpha_paired_changed > 0 or entire_set):
            alpha_paired_changed = 0
            if entire_set:
                for i in range(ss.samples):
                    alpha_paired_changed += SvmComplete.inner_loop(i, ss)
                print "full set, inter:%d, i:%d, pairs changed %d" % (cur_iter, i, alpha_paired_changed)
                cur_iter += 1
            else:
                non_bound_is = np.nonzero((ss.alphas.A > 0) * (ss.alphas.A < ss.uper))[0]
                for i in non_bound_is:
                    alpha_paired_changed += SvmComplete.inner_loop(i, ss)
                    print "non_bound_is, iter: %d i:%d, pairs changed %d" % (cur_iter, i, alpha_paired_changed)
                cur_iter += 1
            if entire_set:
                entire_set = False
            elif alpha_paired_changed == 0:
                entire_set = True
            print "iteration number: %d" % cur_iter
        return ss.intercept, ss.alphas
    @staticmethod
    def get_w(features, labels, alphas):
        """
        get weight of hyper plane
        """
        samples, feature_num = np.shape(features)
        weights = np.zeros((feature_num, 1))
        for i in range(samples):
            weights += np.multiply(alphas[i][0] * labels[0, i], features[i, :].T)
        return weights 
