# coding=utf-8

"""
main function for testing
"""
from model import classifer
from utils import utils
import config
from show import svmplot

if __name__ == "__main__":
    PRO_DIR = config.get_project_dir()
    feature_arr, label_arr = utils.load_data_ml(PRO_DIR + "/data/testSet.txt")
    svm_classifer = classifer.Svm()
    svm_classifer.load_data(feature_arr, label_arr)
    svm_classifer.train()

    svmplot.plot_dots_hyper_lane_2(feature_arr, label_arr, \
        svm_classifer.alphas, svm_classifer.intercept, svm_classifer.weights)
    print "done"

