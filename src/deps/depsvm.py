# conding=utf-8
"""
deps func like plot for svm
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_dots_hyper_lane_2(features, labels, alphas, intercept):
    """
    plot the dots and hyper lane of svm(two dimention)
    """
    # just for test isinstance()
    if isinstance(features, (list)):
        data_arr = np.array(features)
    else:
        data_arr = features
    label_arr = labels
    x = data_arr[:, 0]
    y = data_arr[:, 1]
    x_pos = np.where(label_arr > 0, x, None)
    y_pos = np.where(label_arr > 0, y, None)
    x_neg = np.where(label_arr < 0, x, None)
    y_neg = np.where(label_arr < 0, y, None)
    # print x_pos
    plt.plot(x_pos, y_pos, 'r.')
    plt.plot(x_neg, y_neg, 'b.')
    plt.show()