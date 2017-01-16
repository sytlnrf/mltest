# conding=utf-8
"""
deps func like plot for svm
"""
import numpy as np
import random

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
    f_x_k = np.multiply(ss.alphas, ss.labels).T * (ss.features * ss.features[k, :].T) + ss.intercept
    e_k = f_x_k - ss.labels[k]
    return e_k

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
            e_k = cal_error_k(ss, k)
            delta_e = abs(e_i - e_k)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                e_j = e_k
        return max_k, e_j
    else:
        j = select_j_rand(i, ss.sample_num)
        e_j = cal_error_k(ss, j)

    return j, e_j

def update_e_k(ss, k):
    """
    update error of k
    """
    e_k = cal_error_k(ss, k)
    ss.error_cache[k] = [1, e_k]

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
def inner_loop(i, ss):
    e_i = cal_error_k(ss, i)
    if (ss.labels * e_i < -ss.tolerance and ss.alphas[i] < ss.upter) or \
        (ss.labels[i] * e_i > ss.tolerance and ss.alphas[i] > 0):
        j, e_j = select_j(i, ss, e_i)
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
        ss.alphas[j] = clip_alpha(ss.alphas[j], high, low)
        update_e_k(ss, j)
        if abs(ss.alphas[j] - alpha_j_old) < 0.00001:
            print "j not moving enough"
            return 0
        ss.alphas[i] += ss.labels[j] * ss.labels[i] * (alpha_j_old - ss.alphas[j])
        update_e_k(ss, i)
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
        return 0.0
def smo(features, labels, uper, tolerance, max_iter, k_tube=('lin', 0)):
    ss = smo_struct(features, labels.transpose(), uper, tolerance)
    cur_iter = 0
    entire_set = True
    alpha_paired_changed = 0
    while cur_iter < max_iter and (alpha_paired_changed > 0 or entire_set):
        alpha_paired_changed = 0
        if entire_set:
            for i in range(ss.samples):
                alpha_paired_changed += inner_loop(i, ss)
            print "full set, inter:%d, i:%d, pairs changed %d" % (cur_iter, i, alpha_paired_changed)
            cur_iter += 1
        else:
            non_bound_is = np.nonzero((ss.alphas.A > 0) * (ss.alphas.A < uper))[0]
            for i in non_bound_is:
                alpha_paired_changed += inner_loop(i, ss)
                print "non_bound_is, iter: %d i:%d, pairs changed %d" % (cur_iter, i, alpha_paired_changed)
            cur_iter += 1
        if entire_set:
            entire_set = False
        elif alpha_paired_changed == 0:
            entire_set = True
        print "iteration number: %d" % cur_iter
    return ss.intercept, ss.alphas

def get_w(ss):
    """
    get weight of hyper plane
    """
    features = ss.features
    labels = ss.labels
    alphas = ss.alphas
    samples, feature_num = np.shape(features)
    weights = np.zeros((feature_num, 1))
    for i in range(samples):
        weights += np.multiply(alphas[i][0] * labels[0, i], features[i, :].T)
    return weights   



