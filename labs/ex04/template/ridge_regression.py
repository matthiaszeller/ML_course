# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N, D = tx.shape
    left = tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D)
    right = tx.T.dot(y)
    return np.linalg.solve(left, right)
