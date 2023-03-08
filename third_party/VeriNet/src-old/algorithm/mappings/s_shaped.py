"""
This file contains abstractions for S-shaped activation functions (Sigmoid, Tanh ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch.nn as nn
import numpy as np

from src.algorithm.mappings.abstract_mapping import AbstractMapping


class AbstractSShaped(AbstractMapping):

    """
    An abstract class for S-shaped activation functions.

    Contains functionality common to all S-shaped functions, should be subclassed by the individual functions.
    """

    def __init__(self, num_iter_min_tangent_line: int=2):

        """
        Args:
            num_iter_min_tangent_line   : The number of iterations used in the iterative minimal tangent point method.
        """

        super().__init__()
        self._num_iter_min_tangent_line = num_iter_min_tangent_line

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def is_1d_to_1d(self) -> bool:
        return True

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Defines the torch activation functions abstracted by this class. For example, a ReLU abstraction might have:
        [nn.modules.activation.ReLU, nn.ReLU]

        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return []

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping (by applying the activation function or layer-operation).

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        raise NotImplementedError(f"propagate(...) not implemented in {self.__name__}")

    def split_point(self, xl: float, xu: float) -> float:

        """
        Calculates the preferred split point for branching.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        raise NotImplementedError(f"split_point(...) not implemented in {self.__name__}")

    def linear_relaxation(self,
                          lower_bounds_concrete_in: np.array,
                          upper_bounds_concrete_in: np.array,
                          upper: bool
                          ) -> np.array:

        """
        Calculates the linear relaxation

        The linear relaxation is a Nx2 array, where each row represents a and b of the linear equation:
        l(x) = ax + b

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the nodes
            upper                       : If true, the upper relaxation is calculated, else the lower
        Returns:
            relaxations
        """

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = np.zeros((layer_size, 2))

        # Calculate relaxations where the lower bound is equal to the upper
        self._linear_relaxation_equal_bounds(lower_bounds_concrete_in, upper_bounds_concrete_in, relaxations)

        # Initialize the necessary variables and datastructure
        unequal_bounds_idx = np.argwhere((lower_bounds_concrete_in != upper_bounds_concrete_in)).squeeze()
        unequal_bounds_idx = np.atleast_1d(unequal_bounds_idx)

        mixed_bounds_lower = lower_bounds_concrete_in[unequal_bounds_idx]
        mixed_bounds_upper = upper_bounds_concrete_in[unequal_bounds_idx]
        solved = np.zeros_like(unequal_bounds_idx)

        if upper:
            d_activation = self.derivative(mixed_bounds_upper).squeeze()
            activation = self.propagate(mixed_bounds_lower).squeeze()
        else:
            d_activation = self.derivative(mixed_bounds_lower).squeeze()
            activation = self.propagate(mixed_bounds_upper).squeeze()

        # Try the line intercepting both endpoints
        lines = self._intercept_line(mixed_bounds_lower, mixed_bounds_upper, upper=upper)
        valid = np.argwhere(lines[:, 0] <= d_activation)
        relaxations[unequal_bounds_idx[valid]] = lines[valid]
        solved[valid] = 1

        # Try the optimal tangent line
        lines = self._tangent_line(mixed_bounds_lower, mixed_bounds_upper, upper=upper)
        if upper:
            valid = np.argwhere(lines[:, 0] * mixed_bounds_lower + lines[:, 1] >= activation)
        else:
            valid = np.argwhere(lines[:, 0] * mixed_bounds_upper + lines[:, 1] <= activation)

        relaxations[unequal_bounds_idx[valid]] = lines[valid]
        solved[valid] = 1

        # Use iterative method for the rest
        lines = self._iterative_minimal_tangent_line(mixed_bounds_lower[solved != 1],
                                                     mixed_bounds_upper[solved != 1],
                                                     upper=upper)
        relaxations[unequal_bounds_idx[solved != 1]] = lines

        return relaxations

    def _tangent_line(self,
                      lower_bounds_concrete_in: np.array,
                      upper_bounds_concrete_in: np.array,
                      upper: bool,
                      tangent_point: np.array=None
                      ) -> np.array:

        """
        Calculates the tangent line.

        Args:
            lower_bounds_concrete_in: The array with lower bounds for each node
            upper_bounds_concrete_in: The array with upper bounds for each node
            tangent_point           : The tangent point, if None the optimal tangent point is calculated.
            upper                   : If true, the upper relaxation is calculated, else the lower

        Returns:
            An array where the first column is a and the second column is b, the parameters in l(x) = ax + b

        Note:
            This function does not check that the tangent is a valid bound.
        """

        xu = np.atleast_1d(upper_bounds_concrete_in)
        xl = np.atleast_1d(lower_bounds_concrete_in)

        if tangent_point is None:
            tangent_point = (xu**2 - xl**2)/(2*(xu-xl))  # Optimal tangent point

        act = self.propagate(tangent_point)
        a = self.derivative(tangent_point)
        b = act - a * tangent_point

        # Outward rounding
        num_ops = 11
        max_edge_dist = np.vstack((np.abs(xl), np.abs(xu))).max(axis=0)
        max_err = np.spacing(np.abs(a)) * max_edge_dist + np.spacing(np.abs(b))
        outward_round = max_err * num_ops if upper else - max_err * num_ops
        b += outward_round

        return np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1)

    def _intercept_line(self,
                        lower_bounds_concrete_in: np.array,
                        upper_bounds_concrete_in: np.array,
                        upper: bool) -> np.array:

        """
        Calculates the line intercepting two points of the activation function.

        Args:
            lower_bounds_concrete_in: The array with lower bounds for each node
            upper_bounds_concrete_in: The array with upper bounds for each node
            upper                   : If true, the upper relaxation is calculated, else the lower

        Returns:
            An array where the first column is a and the second column is b, the parameters in l(x) = ax + b
        """

        xu = np.atleast_1d(upper_bounds_concrete_in)
        xl = np.atleast_1d(lower_bounds_concrete_in)

        a = (self.propagate(xu) - self.propagate(xl)) / (xu - xl)
        b = self.propagate(xu) - a * xu

        # Outward rounding
        num_ops = 6
        max_edge_dist = np.vstack((np.abs(xl), np.abs(xu))).max(axis=0)
        max_err = np.spacing(np.abs(a)) * max_edge_dist + np.spacing(np.abs(b))
        outward_round = max_err * num_ops if upper else - max_err * num_ops
        b += outward_round

        return np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1)

    def _iterative_minimal_tangent_line(self,
                                        lower_bounds_concrete_in: np.array,
                                        upper_bounds_concrete_in: np.array,
                                        upper: bool=True) -> np.array:

        """
        Uses the iterative tangent method described in the paper to find the valid tangent line coordinate closest to 0.

        Args:
            lower_bounds_concrete_in: The array with lower bounds for each node
            upper_bounds_concrete_in: The array with upper bounds for each node
            upper                   : If true a upper bound is calculated, else a lower bound is calculated

        Returns:
            An array where the first column is a and the second column is b, the parameters in l(x) = ax + b
        """

        if upper:
            x_bound = lower_bounds_concrete_in
            xi = upper_bounds_concrete_in
        else:
            x_bound = upper_bounds_concrete_in
            xi = lower_bounds_concrete_in

        for i in range(self._num_iter_min_tangent_line):
            xi = self._update_xi(xi, x_bound, upper)

        line = self._tangent_line(lower_bounds_concrete_in, upper_bounds_concrete_in, upper, xi)
        return line

    def derivative(self, x: np.array) -> np.array:

        """
        Calculates the derivative of the activation function at input x.

        Args:
            x: The input
        Returns:
            The derivative of the activation function at x
        """

        raise NotImplementedError("derivative(...) not implemented in subclass")

    def _update_xi(self, xi: np.array, x_bound: np.array, upper: bool):

        """
        Calculates the new xi for the iterative tangent method.

        Args:
            xi              : The last tangent point calculated
            x_bound         : The lower/upper input bound for calculating upper/lower relaxation respectively
            upper           : If true the upper tangent is calculated, else the lower tangent is calculated
        """

        raise NotImplementedError("_update_xi(...) not implemented in subclass")

    def integral(self, xl: np.array, xu: np.array) -> np.array:

        """
        Calculates the integral of the activation function from xl to xu.

        Args:
            xl  : The lower bound
            xu  : The upper bounds
        Returns:
            The integral from xl to xu
        """

        raise NotImplementedError("integral(...) not implemented in subclass")


class Sigmoid(AbstractSShaped):

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.activation.Sigmoid, nn.Sigmoid]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping by applying the Sigmoid element-wise.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array) -> np.array:

        """
        Calculates the derivative of the Sigmoid at input x

        Args:
            x: The input
        Returns:
            The derivative of the Sigmoid at x
        """

        sigmoid = self.propagate(x)
        return sigmoid * (1 - sigmoid)

    def split_point(self, xl: float, xu: float) -> float:

        """
        Returns the preferred split point for the Sigmoid.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point.
        """

        mid = (self.propagate(xu) + self.propagate(xl)) / 2
        return -np.log((1 / mid) - 1)

    def integral(self, xl: np.array, xu: np.array) -> np.array:

        """
        Returns the integral of the Sigmoid from xl to xu.

        Args:
            xl  : The lower bound
            xu  : The upper bounds
        Returns:
            The integral from xl to xu
        """

        return np.log(np.exp(xu) + 1) - np.log(np.exp(xl) + 1)

    def _update_xi(self, xi: np.array, x_bound: np.array, upper: bool):

        """
        Calculates the new xi for the iterative tangent method as described in the paper.

        Args:
            xi              : The last tangent point calculated
            x_bound         : The lower/upper input bound for calculating upper/lower relaxation respectively
            upper           : If true the upper tangent is calculated, else the lower tangent is calculated
        """

        root = (np.sqrt(1 - 4 * (self.propagate(xi) - self.propagate(x_bound))/(xi - x_bound)))/2.
        if upper:
            sxi = 0.5 + root
        else:
            sxi = 0.5 - root
        new_xi = -np.log(1 / sxi - 1)

        return new_xi


class Tanh(AbstractSShaped):

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.activation.Tanh, nn.Tanh]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping by applying the Tanh element-wise

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        return np.tanh(x)

    def derivative(self, x: np.array) -> np.array:

        """
        Calculates the derivative of the Tanh at input x

        Args:
            x: The input
        Returns:
            The derivative of the Tanh at x
        """

        return 1 - self.propagate(x) ** 2

    def split_point(self, xl: float, xu: float) -> float:

        """
        Returns the preferred split point for the Tanh.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        mid = (self.propagate(xu) + self.propagate(xl)) / 2
        return 0.5 * np.log((1 + mid) / (1 - mid))

    def integral(self, xl: np.array, xu: np.array) -> np.array:

        """
        Returns the integral of the Tanh from xl to xu.

        Args:
            xl  : The lower bound
            xu  : The upper bounds
        Returns:
            The integral from xl to xu
        """

        return np.log(0.5*(np.exp(-xu) + np.exp(xu))) - np.log(0.5*(np.exp(-xl) + np.exp(xl)))

    def _update_xi(self, xi: np.array, x_bound: np.array, upper: bool):

        """
        Calculates the new xi for the iterative tangent method as described in the paper.

        Args:
            xi              : The last tangent point calculated
            x_bound         : The lower/upper input bound for calculating upper/lower relaxation respectively
            upper           : If true the upper tangent is calculated, else the lower tangent is calculated
        """

        root = np.sqrt(1 - (self.propagate(xi) - self.propagate(x_bound)) / (xi - x_bound))
        if upper:
            sxi = root
        else:
            sxi = - root
        new_xi = 0.5 * np.log((1 + sxi) / (1 - sxi))

        return new_xi


class SigmoidNaive(AbstractMapping):

    """
    OBS: This class is not used, and only kept for benchmarking purposes, use SigmoidAbstraction() instead.
    """

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

        return []  # Not used, see SigmoidAbstraction() instead.

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping by applying the Sigmoid element-wise.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        return 1 / (1 + np.exp(-x))

    def split_point(self, xl: float, xu: float):

        """
        Returns the preferred split point for branching, which is the input value such that the output is in the middle
        of the minima and maxima.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        mid = (self.propagate(xu) + self.propagate(xl)) / 2
        return -np.log((1 / mid) - 1)

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Calculates the linear relaxation

        The linear relaxation is a Nx2 array, where each row represents a and b of the linear equation:
        l(x) = ax + b.

        The naive relaxations are used, with slope equal to the smallest derivative in the
        input interval.

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the nodes
            upper                       : If true, the upper relaxation is calculated, else the lower
        Returns:
            The relaxations as a Nx2 array
        """

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = np.zeros((layer_size, 2))

        a = np.min(np.hstack((self.derivative(lower_bounds_concrete_in).reshape((-1, 1)),
                              self.derivative(upper_bounds_concrete_in).reshape((-1, 1)))), axis=1)

        if upper:
            b = self.propagate(upper_bounds_concrete_in) - a * upper_bounds_concrete_in
        else:
            b = self.propagate(lower_bounds_concrete_in) - a * lower_bounds_concrete_in

        relaxations[:, 0] = a
        relaxations[:, 1] = b

        return relaxations

    def derivative(self, x: np.array) -> np.array:

        """
        Calculates the derivative of the Sigmoid at input x

        Args:
            x: The input
        Returns:
            The derivative of the Sigmoid at x
        """

        sigmoid = self.propagate(x)
        return np.array(sigmoid * (1 - sigmoid))

    # noinspection PyMethodMayBeStatic
    def integral(self, xl: np.array, xu: np.array) -> np.array:

        """
        Returns the integral of the Sigmoid from xl to xu.

        Args:
            xl  : The lower bound
            xu  : The upper bounds
        Returns:
            The integral from xl to xu
        """

        return np.log(np.exp(xu) + 1) - np.log(np.exp(xl) + 1)
