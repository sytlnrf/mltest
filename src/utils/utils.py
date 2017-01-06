# coding=utf-8
"""
util functions
"""
import numpy as np

def load_data_ml(file_name):
    """
    load data from txt format file
    :param file_name: string
        file path(include full name)
        file contents like:
            3.542485	1.977398	-1
            3.018896	2.556416	-1
            7.551510	-1.580030	1
            2.114999	-0.004466	-1
            8.127113	1.274372	1
    :returns:
        features:
            [
                [v11, v12, v13, ......, v1n],
                [v21, v22, v23, ......, v2n],
                .
                .
                [vm1, vm2, vm3, ......, vmn]
            ]
        labels:
            [1, 1, -1, ......, 1]
            length = m
    """
    features = np.array()
    labels = np.array()
    file_handle = open(file_name)
    for line in file_handle.readlines():
        line_arr = line.strip().split('\t')
        np.append(features, line_arr[:-1])
        np.append(labels, float(line_arr[-1]))
    return features, labels
