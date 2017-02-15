# coding=utf-8
"""
Author: shengyitao
Create Date: 2017-02-12
"""

import numpy as np
import math

class AdaboostDemo(object):
    """
    demo for adaboost algorithm
    """
    def __init__(self):
        self.feats = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        self.labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
        self.sample_num = np.shape(self.feats)[0]
        self.alphas = None
        self.train()
        for ite in range(self.sample_num):
            print self.ensemble_classifer(self.feats[ite][0])
    def train(self):
        """
        train the weak classifers and get the ensemble method(adaboost algorithm)
        """
        # init weights
        weights = np.ones(self.sample_num) / self.sample_num
        alphas = []
        for ite_cls in range(3):
            err_total = 0.0
            z_totoal = 0.0
            cls_result = np.array([])
            for ite in range(self.sample_num):
                feat_value = self.feats[ite][0]
                g_cls = self.classifer(ite_cls, feat_value)
                if g_cls != self.labels[ite]:
                    err_total += weights[ite]
                cls_result = np.append(cls_result, g_cls)
            alpha = np.log((1.0 - err_total) / err_total) / 2.0
            tmp_result = np.exp(-alpha * np.array(self.labels) * cls_result)
            z_totoal = np.dot(weights, tmp_result)
            weights = weights * tmp_result / z_totoal
            alphas.append(alpha)

        self.alphas = np.array(alphas)

    def ensemble_classifer(self, feat_value):
        """
        ensemble classifer
        """
        alphas = self.alphas
        ensem = np.dot(alphas, np.array([self.classifer(0, feat_value), \
            self.classifer(1, feat_value), self.classifer(2, feat_value)]))
        return ensem

    def classifer(self, c_id, feat_value):
        """
        single classifer
        :paras c_id:
            type: integer
            classifer id to choose which classifer to use
        """
        # demo classifer threshold
        thres = [2.5, 8.5, 5.5]
        # classifer logistic
        if c_id < 2:
            if feat_value < thres[c_id]:
                return 1
            else:
                return -1
        elif c_id == 2:
            if feat_value < thres[c_id]:
                return -1
            else:
                return 1
        else:
            return -2
    class Adaboost(object):
        def __init__(self, feat_values, labels):
            print "init"
        @staticmethod
        def stump_classifier(feat_values, dimen, thres_val, thres_ineq):
            """
            """
            ret_array = np.ones((np.shape(feat_values)[0], 1))
            if thres_ineq == 'lt':
                ret_array[feat_values[: dimen] <= thres_val] = -1
            else:
                ret_array[feat_values[:, dimen] > thres_val] = -1
            return ret_array

        @staticmethod
        def build_stump(feat_values, labels, weight_dis):
            """
            build the stump classifer
            :params feat_value: feature values matrix
            :params labels: class label of each sample
            :params weight_dis: initial weights distribution of adaboost
            """
            m, n = np.shape(feat_values)
            step_num = 10
            final_stump = {}
            best_class_est = np.matrix(np.zeros((m, 1)))
            min_err = float("inf")
            for ite in range(n):
                min_value = feat_values[:, ite].min()
                max_value = feat_values[:, ite].max()
                step_size = (max_value - min_value) / float(step_num)
                temp_step = step_num + 1
                for ite_value in range(-1, temp_step):
                    for inequal in ['lt', 'gt']:
                        thres_value = (min_value + float(ite_value) * step_size)
                        predict_value = Adaboost.stump_classifier(feat_values, thres_value, inequal)
                        err_arr = np.matrix(np.ones((m, 1)))
                        err_arr[predict_value == labels] = 0
                        weight_err = weight_dis.T * err_arr
                        print "split: dim %d, thres %.2f, thres \
                            inequal: %s, the weight error is %.3f" %\
                            (ite, thres_value, inequal, weight_err)
                        if weight_err < min_err:
                            min_err = weight_err
                            best_class_est = predict_value.copy()
                            final_stump['dim'] = ite
                            final_stump['thres'] = thres_value
                            final_stump['inequal'] = inequal
            return final_stump, min_err, best_class_est

if __name__ == "__main__":
    # demoobj = AdaboostDemo()