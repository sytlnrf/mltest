# coding=utf-8

"""
main function for testing
"""
from model import classifer
from utils import utils
import config
from show import svmplot
import numpy as np
from sklearn import svm
from model import regression
import matplotlib.pyplot as plt

def test_inte_svm():
    PRO_DIR = config.get_project_dir()
    feature_arr, label_arr = utils.load_text_ml(PRO_DIR + "/data/abalone.txt")
    # feature_arr, label_arr = utils.load_csv_ml(PRO_DIR + "/data/index_000300_daily.csv")
    clf = svm.SVC()
    clf.fit(feature_arr, label_arr)
    print clf.class_weight_, clf._get_coef()
    # print clf.support_vector_
if __name__ == "__main__":
    PRO_DIR = config.get_project_dir()
    feature_arr, label_arr = utils.load_text_ml(PRO_DIR + "/data/abalone.txt")
    # feature_arr, label_arr = utils.load_csv_ml(PRO_DIR + "/data/index_000300_daily.csv")
    # print np.shape(feature_arr), np.shape(label_arr)
    # svm_classifer = classifer.SvmComplete(feature_arr, label_arr, 0.6, 0.001)
    # svm_classifer.fit()
    # svm_classifer.load_data(feature_arr, label_arr)
    # svm_classifer.train()
    # svmplot.plot_dots_hyper_lane_2(feature_arr, label_arr, \
    #     np.asmatrix(np.zeros((199, 1))), np.asmatrix([[-1.0]]), np.asmatrix([[0.1], [0.2]]))
    # svmplot.plot_dots_hyper_lane_2(feature_arr, label_arr, \
    #     svm_classifer.alphas, svm_classifer.intercept, svm_classifer.weights)
    # feature_arr, label_arr = utils.load_csv_ml(PRO_DIR + "/data/index_000300_daily.csv")

    # test_inte_svm()
    ####################
    # regression.stand_regress(feature_arr, label_arr)
    # print regression.lwlr(feature_arr[0], feature_arr, label_arr, k=0.001)
    # print feature_arr
    # y1 = regression.lwlr_test(feature_arr[0:99], feature_arr[0:99], label_arr[0:99], k=0.1)
    # print regression.res_error(label_arr[0:99],y1)
    w = regression.ridgeTest(feature_arr, label_arr)
    print w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w)
    plt.show()






