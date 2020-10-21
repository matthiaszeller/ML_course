# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Least squares regression using normal equations.

    :return: (w, loss), last weight vector and corresponding loss value."""
    # We use `linalg.solve` instead of matrix inversion (`linalg.inv`) to increase
    # robustness against an ill-conditionned matrix `tx`.
    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w_star
