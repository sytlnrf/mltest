# coding=utf-8
"""
util functions
"""
import numpy as np
import pandas as pd 

def load_text_ml(file_name):
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
    features = []
    labels = []
    file_handle = open(file_name)
    for line in file_handle.readlines():
        line_arr = line.strip().split('\t')
        features.append(line_arr[:-1])
        labels.append(line_arr[-1])
    features = np.array(features)
    labels = np.array(labels)
    # features_mat = np.asmatrix(features).astype(float)
    # labels_mat = np.asmatrix(labels).astype(float)
    return features, labels

def load_csv_ml(file_name):
    """
    load stock data from csv format file
    :param file_name: string
        file path(include full name)
        file contents like:
            date        close       open
            2010-01-04,3535.229,3592.468
            2010-01-05,3564.038,3545.186
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
    csv_data = pd.read_csv(file_name, index_col=['date'], parse_dates=['date'])
    retrive_data = csv_data[['close', 'open', 'ratio']][:200]
    retrive_data['ratio'] = np.where(retrive_data['ratio'] < 0, -1, 1)
    data_array = retrive_data.values
    rows, columns = np.shape(data_array)
    del columns
    features = data_array[:rows-1, (0, 1)]
    labels = data_array[1:, 2]
    features_mat = np.asmatrix(features).astype(float)
    labels_mat = np.asmatrix(labels).astype(float)
    return features_mat, labels_mat
