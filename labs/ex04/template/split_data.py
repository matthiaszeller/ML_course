# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    :return: (xtrain, ytrain, xtest, ytest)
    """
    N = len(x)
    # Shuffle indices
    np.random.seed(seed)
    shuffling_indices = np.random.permutation(N)
    x = x[shuffling_indices]
    y = y[shuffling_indices]
    # Split index
    sid = ceil(N * ratio)

    return x[:sid], y[:sid], x[sid:], y[sid:]

