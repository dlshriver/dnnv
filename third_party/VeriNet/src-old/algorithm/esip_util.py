
"""
Util functions for nn_bounds

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from enum import Enum
import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def concretise_symbolic_bounds_jit(input_bounds: np.array, symbolic_bounds: np.array, outward_round: float=0):

    """
    Calculates the concrete input bounds from the symbolic.

    Args:
        input_bounds    : The input bounds
        symbolic_bounds : The symbolic bounds
        outward_round   : If >0, each float operation in the concretization is rounded outwards by this amount
    """

    layer_size = symbolic_bounds.shape[0]
    concrete_bounds_in = np.empty((layer_size, 2))

    for j in prange(layer_size):

        # Add bias
        temp_lower = symbolic_bounds[j, -1]
        temp_upper = symbolic_bounds[j, -1]

        for i in range(input_bounds.shape[0]):

            coeff = symbolic_bounds[j, i]

            if coeff < 0:
                temp_lower += coeff * input_bounds[i, 1] - outward_round
                temp_upper += coeff * input_bounds[i, 0] + outward_round
            else:
                temp_lower += coeff * input_bounds[i, 0] - outward_round
                temp_upper += coeff * input_bounds[i, 1] + outward_round

        concrete_bounds_in[j] = (temp_lower, temp_upper)

    return concrete_bounds_in


@jit(nopython=True, cache=True)
def sum_error_jit(error_matrix: np.array):

    """
    Calculates the lower and upper error for each node.

    Args:
        error_matrix    : The error matrix where each row is a node in the current layer and each column is a
                          node in the previous layers
    """

    layer_size = error_matrix.shape[0]
    concrete_error = np.empty((layer_size, 2))

    for j in range(layer_size):

        temp_error_lower = 0
        temp_error_upper = 0

        for error in error_matrix[j, :]:

            if error < 0:
                temp_error_lower += error
            else:
                temp_error_upper += error

        concrete_error[j] = temp_error_lower, temp_error_upper

    return concrete_error
