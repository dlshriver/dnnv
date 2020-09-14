"""
This file contains abstractions for layers (FC, Conv ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf

from src.algorithm.mappings.abstract_mapping import AbstractMapping, \
    ActivationFunctionAbstractionException


class FC(AbstractMapping):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.linear.Linear]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping by applying the fully-connected mapping.

        Args:
            x           : The input as a np.array.
                          Assumed to be a NxM vector where the rows represent nodes and the columns represent
                          coefficients of the symbolic bounds. Can be used on concrete values instead of equations by
                          shaping them into an Nx1 array.
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        x = self.params["weight"] @ x

        if add_bias:
            x[:, -1] += self.params["bias"]

        return x

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Not implemented since function is linear.
        """

        msg = f"linear_relaxation(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    # noinspection PyTypeChecker
    def split_point(self, xl: float, xu: float) -> float:

        """
        Not implemented since function is linear.
        """

        msg = f"split_point(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    def out_shape(self, in_shape: np.array) -> np.array:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return np.array((self.params["weight"].shape[0]))


class Conv2d(AbstractMapping):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias", "kernel_size", "padding", "stride", "in_channels", "out_channels"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.Conv2d]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping (by applying the activation function or layer-operation).

        Args:
            x           : The input as a np.array.
                          Assumed to be a NxM vector where the rows represent nodes and the columns represent
                          coefficients of the symbolic bounds. Can be used on concrete values instead of equations by
                          shaping them into an Nx1 array.
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        stride = self.params["stride"]
        padding = self.params["padding"]
        weights = torch.Tensor(self.params["weight"])
        bias = torch.Tensor(self.params["bias"])
        in_shape = self.params["in_shape"]
        out_size = np.prod(self.out_shape(in_shape))

        # Reshape to 2d, stacking the coefficients of the symbolic equation in dim 0, the "batch" dimension"
        x_2d = torch.Tensor(x.T.reshape((-1, *in_shape)))

        # Perform convolution on the reshaped input
        y_2d = tf.conv2d(x_2d, weight=weights, stride=stride, padding=padding)

        # Add the bias to the last "batch" dimension, since this is the constant value of the equations
        if add_bias:
            y_2d[-1, :, :, :] += bias.view(-1, 1, 1)

        # Reshape to NxM shaped where N are the nodes and M are the coefficients for the equations
        y = y_2d.detach().numpy().reshape(-1, out_size).T

        return y

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Not implemented since function is linear.
        """

        msg = f"linear_relaxation(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    # noinspection PyTypeChecker
    def split_point(self, xl: float, xu: float) -> float:

        """
        Not implemented since function is linear.
        """

        msg = f"split_point(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    def out_shape(self, in_shape: np.array) -> np.array:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        params = self.params
        channels = params["out_channels"]
        height = (in_shape[1] - params["kernel_size"][0] + 2 * params["padding"][0]) // params["stride"][0] + 1
        width = (in_shape[2] - params["kernel_size"][1] + 2 * params["padding"][1]) // params["stride"][1] + 1

        return np.array((channels, height, width))


class BatchNorm2d(AbstractMapping):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias", "running_mean", "running_var", "eps"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.BatchNorm2d]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping (by applying the activation function or layer-operation).

        Args:
            x           : The input as a np.array.
                          Assumed to be a NxM vector where the rows represent nodes and the columns represent
                          coefficients of the symbolic bounds. Can be used on concrete values instead of equations by
                          shaping them into an Nx1 array.
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        if x.shape[0] == 0:
            return x

        std = np.sqrt(self.params["running_var"] + self.params["eps"])
        std = std.reshape(-1, 1, 1)
        mean = self.params["running_mean"].reshape(-1, 1, 1)
        gamma = self.params["weight"].reshape(-1, 1, 1)
        beta = self.params["bias"].reshape(-1, 1, 1)
        in_shape = self.params["in_shape"]
        out_size = np.prod(self.out_shape(in_shape))

        # Reshape to 2d, stacking the coefficients of the symbolic equation in dim 0, the "batch" dimension"
        x_2d = x.T.reshape((-1, *(in_shape)))

        # Calculate batch-normalisation
        y_2d = x_2d.copy()
        if add_bias:
            y_2d[-1, :, :, :] -= mean
        y_2d = (y_2d / std) * gamma
        if add_bias:
            y_2d[-1, :, :, :] += beta

        # Reshape back to 1d
        y_2d = y_2d.reshape(-1, out_size).T

        return y_2d

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Not implemented since function is linear.
        """

        msg = f"linear_relaxation(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    # noinspection PyTypeChecker
    def split_point(self, xl: float, xu: float) -> float:

        """
        Not implemented since function is linear.
        """

        msg = f"split_point(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)
