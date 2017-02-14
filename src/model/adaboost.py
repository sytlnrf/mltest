# coding=utf-8
"""
Author: shengyitao
Create Date: 2017-02-12
"""

import numpy as np

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
if __name__ == "__main__":
    demoobj = AdaboostDemo()