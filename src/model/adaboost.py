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

    def train(self):
        """
        train the weak classifers and get the ensemble method(adaboost algorithm)
        """
        # init weights
        weights = np.ones(self.sample_num) / self.sample_num

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
    demoobj.train()