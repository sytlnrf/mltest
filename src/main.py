# coding=utf-8

"""
main function for testing
"""
from model import classifer
from utils import utils
import config

if __name__ == "__main__":
    pro_dir = config.get_project_dir()
    feature_arr, label_arr = utils.load_data_ml(pro_dir + "/data/testSet.txt")
    print feature_arr

