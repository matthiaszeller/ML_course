# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    power_mx = np.tile(np.arange(degree + 1), (len(x), 1))
    augmented_x = np.tile(np.array(x).reshape(-1, 1), (1, degree + 1))
    return np.power(augmented_x, power_mx)
