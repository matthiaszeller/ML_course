# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    """Compute the mean squared error (MAE) loss value:
    MSE(w) = 1/2N * sum{i = 1 to n} (y_n - x_n^T * w)^2
           = 1/2N * e^T * e

    :param y: response vector
    :param tx: feature matrix extended with 1's as 1st column"""
    # TODO: decide if best to implement MSE formula with  (1/N) or (1/2N <- chosen yet))
    e = y - np.matmul(tx, w)
    return np.inner(e, e) / len(y) / 2


def least_squares(y, tx):
    """Least squares regression using normal equations.

    :return: (w, loss), last weight vector and corresponding loss value."""
    # We use `linalg.solve` instead of matrix inversion (`linalg.inv`) to increase
    # robustness against an ill-conditionned matrix `tx`.
    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w_star)
    return w_star, loss
