# coding=utf-8
import numpy as np

def stand_regress(feat_values, labels):
    x_mat = np.mat(feat_values,dtype=float)
    y_mat = np.mat(labels, dtype=float).T
    x_T_x = x_mat.T * x_mat
    if np.linalg.det(x_T_x) == 0.0:
        print "can not do inverse"
    w_s = x_T_x.I * (x_mat.T * y_mat) 
    return w_s
def lwlr(x_i, feat_values, labels, k=1.0):
    x_i = np.mat(x_i, dtype=float)[0]
    x_mat = np.mat(feat_values,dtype=float)
    y_mat = np.mat(labels, dtype=float).T
    m = np.shape(x_mat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff_mat = x_i - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k**2))
    x_T_x = x_mat.T * (weights * x_mat)
    if np.linalg.det(x_T_x) == 0.0:
        print "can not do inverse"
        return
    w_s = x_T_x.I * (x_mat.T * (weights * y_mat))
    # print w_s
    return x_i * w_s
def lwlr_test(test_data, feat_values, labels, k=1.0):
    m = np.shape(test_data)[0]
    y_heris = np.zeros(m)
    for i in range(m):
        y_heris[i] = lwlr(test_data[i], feat_values, labels, k)
    return y_heris
def ridge_regress(feat_values, labels, lam=0.2):
    x_T_x = feat_values.T * feat_values
    denom = x_T_x + np.eye(np.shape(feat_values)[1]) * lam
    if np.linalg.det(denom) == 0:
        print "can not do inverse"
        return
    ws = denom.I * (feat_values.T * labels)
    return ws
def ridgeTest(feat_values, labels):
    x_mat = np.mat(feat_values, dtype=float)
    y_mat = np.mat(labels, dtype=float).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regress(x_mat, y_mat, np.exp(i-10))
        w_mat[i, :] = ws.T
    return w_mat
def res_error(y1, y2):
    y_r = np.array(y1, dtype=float)
    y_t = np.array(y2, dtype=float)
    print y_r
    return ((y_r - y_t)**2).sum()
# if __name__ == "__main__":
#     feat_values, labels = 