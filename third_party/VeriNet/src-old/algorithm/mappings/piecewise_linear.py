"""
This file contains abstractions for piecewise linear activation functions (ReLU, Identity ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch.nn as nn
import numpy as np

from src.algorithm.mappings.abstract_mapping import AbstractMapping, \
    ActivationFunctionAbstractionException


class Relu(AbstractMapping):

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def is_1d_to_1d(self) -> bool:
        return True

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.activation.ReLU, nn.ReLU]

    def propagate(self, x: np.array, add_bias: bool = True):

        """
        Propagates trough the mapping by applying the ReLU function element-wise.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        return (x > 0) * x

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Calculates the linear relaxation

        The linear relaxation is a Nx2 array, where each row represents a and b of the linear equation:
        l(x) = ax + b

        The relaxations are described in detail in the paper.

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the nodes
            upper                       : If true, the upper relaxation is calculated, else the lower
        Returns:
            The relaxations as a Nx2 array
        """

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = np.zeros((layer_size, 2))

        # Operating in the positive area
        fixed_upper_idx = np.argwhere(lower_bounds_concrete_in >= 0)
        relaxations[fixed_upper_idx, 0] = 1
        relaxations[fixed_upper_idx, 1] = 0

        # Operating in the negative area
        fixed_lower_idx = np.argwhere(upper_bounds_concrete_in <= 0)
        relaxations[fixed_lower_idx, :] = 0

        # Operating in the non-linear area
        mixed_idx = np.argwhere((upper_bounds_concrete_in > 0)*(lower_bounds_concrete_in < 0))

        if len(mixed_idx) == 0:
            return relaxations

        xl = lower_bounds_concrete_in[mixed_idx]
        xu = upper_bounds_concrete_in[mixed_idx]

        if upper:
            a = xu / (xu - xl)
            b = - a * xl
            relaxations[:, 0][mixed_idx] = a
            relaxations[:, 1][mixed_idx] = b
        else:
            relaxations[:, 0][mixed_idx] = xu / (xu - xl)
            relaxations[:, 1][mixed_idx] = 0

        # Outward rounding
        num_ops = 2
        max_edge_dist = np.hstack((np.abs(xl), np.abs(xu))).max(axis=1)[:, np.newaxis]
        max_err = (np.spacing(np.abs(relaxations[:, 0][mixed_idx])) * max_edge_dist
                   + np.spacing(np.abs(relaxations[:, 1][mixed_idx])))
        outward_round = max_err * num_ops if upper else - max_err * num_ops
        relaxations[:, 1][mixed_idx] += outward_round

        return relaxations

    def split_point(self, xl: float, xu: float):

        """
        Returns the preferred split point for branching which is 0 for the ReLU.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        return 0


class Identity(AbstractMapping):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return True

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping by returning the input unchanged.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        return x

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        This function is used to create a mapping from torch functions to their abstractions.

        Returns:
           A list with all torch functions that are abstracted by the current subclass.
        """

        return []

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Not implemented since function is linear.
        """

        msg = f"linear_relaxation(...) not implemented for {self.__name__} since it is linear "
        raise ActivationFunctionAbstractionException(msg)

    def split_point(self, xl: float, xu: float):

        """
        Not implemented since function is linear.
        """

        msg = f"split_point(...) not implemented for {self.__name__} since it is linear "
        raise ActivationFunctionAbstractionException(msg)
