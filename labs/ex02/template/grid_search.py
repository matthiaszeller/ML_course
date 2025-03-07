# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs


def generate_w(num_intervals, low=(-100, -150), up=(200, 150)):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(low[0], up[0], num_intervals)
    w1 = np.linspace(low[1], up[1], num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************
