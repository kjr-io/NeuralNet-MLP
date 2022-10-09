import numpy as np
'''
Three Functions to Build the Datasets for the XOR, OR, and AND Gates
'''
def xor_data():
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    return x_train, y_train

def or_data():
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[1]]])
    return x_train, y_train

def and_data():
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[0]], [[0]], [[1]]])
    return x_train, y_train


