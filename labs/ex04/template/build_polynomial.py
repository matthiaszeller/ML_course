# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """Polynomial expansion of feature vector `x` up to degree `degree`.

    :param x: 1D array or list representing a feature vector
    :param int degree:
    :return: 2D array of shape `len(x) * (degree+1)`"""
    # Build matrix of powers (columns of 1's, 2's, ..., (degree+1)'s)
    power_mx = np.tile(np.arange(degree + 1), (len(x), 1))
    # Build matrix whose columns are duplicated x's arrays
    augmented_x = np.tile(np.array(x).reshape(-1, 1), (1, degree + 1))
    # Raise features to powers of `power_mx`, element-wise
    return np.power(augmented_x, power_mx)
