# coding=utf-8
import numpy as np

def stand_regress(feat_values, labels):
    x_mat = np.mat(feat_values,dtype=float)
    y_mat = np.mat(labels, dtype=float).T
    print x_mat
    x_T_x = x_mat.T * x_mat
    if np.linalg.det(x_T_x) == 0.0:
        print "can not do inverse"
    w_s = x_T_x.I * (x_mat.T * y_mat)
    return w_s
# if __name__ == "__main__":
#     feat_values, labels = 