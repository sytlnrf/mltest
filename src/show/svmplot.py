# conding=utf-8
"""
deps func like plot for svm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_dots_hyper_lane_2(features, labels, alphas, intercept, weights):
    """
    plot the dots and hyper lane of svm(two dimention)
    """
    # just for test isinstance()
    if isinstance(features, (list)):
        data_arr = np.array(features).astype(float)
    else:
        data_arr = features.astype(float)
    label_arr = labels.astype(float)
    x = data_arr[:, 0]
    y = data_arr[:, 1]

    x_pos = np.where(label_arr > 0, x, None)
    y_pos = np.where(label_arr > 0, y, None)
    x_neg = np.where(label_arr < 0, x, None)
    y_neg = np.where(label_arr < 0, y, None)
    # print alphas


    suport_vector = data_arr[np.nonzero(alphas)[0]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_pos, y_pos, marker='s', s=90)
    ax.scatter(x_neg, y_neg, marker='o', s=50, c='red')
    for svc in suport_vector:
        # circle=plt.Circle(svc,2)
        circle = Circle(svc, 0.25, \
            facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    # ax.plot(x,y)
    svc_x = np.arange(-2.0, 12.0, 0.1)
    svc_y = (-weights[0][0] * svc_x - intercept) / weights[1][0]
    # print svc_x
    svc_y = np.asarray(svc_y)[0]
    ax.plot(svc_x, svc_y)
    ax.axis([-2, 12, -8, 6])
    plt.show()
    # plt.plot(x_pos, y_pos, 'r.')
    # plt.plot(x_neg, y_neg, 'b.')
    # plt.show()