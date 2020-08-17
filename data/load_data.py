import numpy as np
import os
import csv

def load_proj_helper(count, path):
    """
    helper for load_proj_data

    Arguments:
        count - (int) The number of samples to load.
        path - (string) The path to the csv file to load from.

    Returns:
        X - (np.array) An Nxd array where N is count and d is the number of features
        Y - (np.array) A 1D array of targets of size N
    """
    X = np.empty((count, 3))
    Y = np.empty(count)
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        i = 0
        for line in csv_reader:
            if i >= count:
                break
            X[i, 0] = line[0]
            X[i, 1] = line[3]
            X[i, 2] = line[5]
            if line[8] == "Casual":
                Y[i] = 0
            else:
                Y[i] = 1
            i += 1
    return X, Y

def load_proj_data(train_count, test_count, base_folder='data'):
    """
    Loads biker data

    Arguments:
        train_count - (int) The number of training samples to load.
        test_count - (int) The number of testing samples to load.
        base_folder - (string) path to dataset.

    Returns:
        trainX - (np.array) An Nxd array of features, where N is train_count and
            d is the number of features.
        testX - (np.array) An Mxd array of  features, where M is test_count and
            d is the number of features.
        trainY - (np.array) A 1D array of targets of size N.
        testY - (np.array) A 1D array of targets of size M.
    """
    train_path = os.path.join(base_folder, '2017Q1-capitalbikeshare-tripdata.csv')
    test_path = os.path.join(base_folder, '2017Q2-capitalbikeshare-tripdata.csv')
    trainX, trainY = load_proj_helper(train_count, train_path)
    testX, testY = load_proj_helper(test_count, test_path)
    return trainX, testX, trainY, testY
