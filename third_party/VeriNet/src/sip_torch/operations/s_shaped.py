"""
This file contains abstractions for S-shaped activation functions (Sigmoid, Tanh ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn

from verinet.sip_torch.operations.abstract_operation import AbstractOperation


def atleast_1d(x: torch.Tensor):

    if x.dim() == 0:
        return x.reshape(1)
    else:
        return x


class AbstractSShaped(AbstractOperation):

    """
    An abstract class for S-shaped activation functions.

    Contains functionality common to all S-shaped functions, should be subclassed by
    the individual functions.
    """

    def __init__(self, num_iter_min_tangent_line: int = 2):

        """
        Args:
            num_iter_min_tangent_line:
                The number of iterations used in the iterative minimal tangent point
                method.
        """

        super().__init__()
        self._num_iter_min_tangent_line = num_iter_min_tangent_line

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Defines the torch activation functions abstracted by this class. For example,
        a ReLU abstraction might have: [nn.modules.activation.ReLU, nn.ReLU].

        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return []

    def get_num_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> int:

        """
        Returns the number of non-linear neurons based on bounds.

        Args:
            bounds_concrete_pre:
                A Nx2 tensor with the lower bounds of the neurons in the first col
                and upper bounds in the second.
        """

        return bounds_concrete_pre.shape[0]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    def ssip_forward(self, bounds_symbolic_pre: torch.Tensor, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower bounding equations of ssip.

        Args:
            bounds_symbolic_pre:
                A 2xNxM tensor with the symbolic bounds. The first dimension contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        raise NotImplementedError(f"propagate_reversed(...) not implemented")

    def propagate_const_term(self, const_terms: torch.Tensor) -> torch.Tensor:

        """
        Propagates the constant part of the symbolic bound for ssip.

        Args:
            const_terms:
                A Nx2 tensor with the constant part of the lower equation in the first
                column and the upper in the second.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    def split_point(self, xl: float, xu: float) -> float:

        """
        Calculates the preferred split point for branching.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point.
        """

        raise NotImplementedError(f"split_point(...) not implemented")

    def linear_relaxation(self,
                          lower_bounds_concrete_in: torch.Tensor,
                          upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False
                          ) -> torch.Tensor:

        """
        Calculates the linear relaxation.

        The linear relaxation is a 2xNx2 tensor, where relaxation[0] represents a and b
        of the lower relaxation equation: l(const_terms) = ax + b (correspondingly,
        relaxation[1] is the upper relaxation).

        The relaxations are described in detail in the VeriNet paper.

        Args:
            lower_bounds_concrete_in:
                The concrete lower bounds of the input to the nodes.
            upper_bounds_concrete_in:
                The concrete upper bounds of the input to the nodes.
            force_parallel:
                Not implemented
        Returns:
            The relaxations.
        """

        lower_relaxation = self._linear_relaxation_single(lower_bounds_concrete_in, upper_bounds_concrete_in,
                                                          upper=False)
        upper_relaxation = self._linear_relaxation_single(lower_bounds_concrete_in, upper_bounds_concrete_in,
                                                          upper=True)

        if force_parallel:

            lower_dist = (upper_relaxation[:, 0] * lower_bounds_concrete_in + upper_relaxation[:, 1]) - \
                         (lower_relaxation[:, 0] * lower_bounds_concrete_in + lower_relaxation[:, 1])
            upper_dist = (upper_relaxation[:, 0] * upper_bounds_concrete_in + upper_relaxation[:, 1]) - \
                         (lower_relaxation[:, 0] * upper_bounds_concrete_in + lower_relaxation[:, 1])

            dists = torch.cat((lower_dist.unsqueeze(1), upper_dist.unsqueeze(1)), dim=1)
            max_dist = torch.max(dists, dim=1)[0]

            lower_relaxation = upper_relaxation.clone()
            lower_relaxation[:, 1] -= max_dist

        relaxations = torch.cat((lower_relaxation.unsqueeze(0), upper_relaxation.unsqueeze(0)))

        return relaxations

    # noinspection PyTypeChecker
    def _linear_relaxation_single(self,
                                  lower_bounds_concrete_in: torch.Tensor,
                                  upper_bounds_concrete_in: torch.Tensor,
                                  upper: bool
                                  ) -> torch.Tensor:

        """
        Calculates the lower or upper linear relaxation depending on the value of
        'upper'.

        The linear relaxation is a Nx2 tensor, where each row represents a and b of the
        linear equation: l(x) = ax + b.

        Args:
            lower_bounds_concrete_in:
                The concrete lower bounds of the input to the nodes.
            upper_bounds_concrete_in:
                The concrete upper bounds of the input to the nodes.
            upper:
                If true, the upper relaxation is calculated, else the lower.
        Returns:
            The relaxations.
        """

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = torch.zeros((layer_size, 2), dtype=self._precision).to(device=lower_bounds_concrete_in.device)

        # Calculate relaxations where the lower bound is equal to the upper
        self._linear_relaxation_equal_bounds(lower_bounds_concrete_in, upper_bounds_concrete_in, relaxations)

        # Initialize the necessary variables and datastructure
        unequal_bounds_idx = torch.nonzero((lower_bounds_concrete_in != upper_bounds_concrete_in)).squeeze()
        unequal_bounds_idx = atleast_1d(unequal_bounds_idx)

        mixed_bounds_lower = lower_bounds_concrete_in[unequal_bounds_idx]
        mixed_bounds_upper = upper_bounds_concrete_in[unequal_bounds_idx]
        solved = torch.zeros_like(unequal_bounds_idx)

        if upper:
            d_activation = self.derivative(mixed_bounds_upper).squeeze()
            activation = self.forward(mixed_bounds_lower).squeeze()
        else:
            d_activation = self.derivative(mixed_bounds_lower).squeeze()
            activation = self.forward(mixed_bounds_upper).squeeze()

        # Try the line intercepting both endpoints
        lines = self._intercept_line(mixed_bounds_lower, mixed_bounds_upper)
        valid = torch.nonzero(lines[:, 0] <= d_activation)
        relaxations[unequal_bounds_idx[valid]] = lines[valid]
        solved[valid] = 1

        # Try the optimal tangent line
        lines = self._tangent_line(mixed_bounds_lower, mixed_bounds_upper)
        if upper:
            valid = torch.nonzero(lines[:, 0] * mixed_bounds_lower + lines[:, 1] >= activation)
        else:
            valid = torch.nonzero(lines[:, 0] * mixed_bounds_upper + lines[:, 1] <= activation)

        relaxations[unequal_bounds_idx[valid]] = lines[valid]
        solved[valid] = 1

        # Use iterative method for the rest
        lines = self._iterative_minimal_tangent_line(mixed_bounds_lower[solved != 1],
                                                     mixed_bounds_upper[solved != 1],
                                                     upper=upper)
        relaxations[unequal_bounds_idx[solved != 1]] = lines

        return relaxations

    # noinspection DuplicatedCode
    def _tangent_line(self,
                      lower_bounds_concrete_in: torch.Tensor,
                      upper_bounds_concrete_in: torch.Tensor,
                      tangent_point: torch.Tensor = None,
                      ) -> torch.Tensor:

        """
        Calculates the tangent line.

        Args:
            lower_bounds_concrete_in:
                The array with lower bounds for each neuron_num.
            upper_bounds_concrete_in:
                The array with upper bounds for each neuron_num.
            tangent_point:
                The tangent point, if None the optimal tangent point is calculated.

        Returns:
            An array where the first column is a and the second column is b, the
            parameters in l(x) = ax + b.

        Note:
            This function does not check that the tangent is a valid bound.
        """

        xu = atleast_1d(upper_bounds_concrete_in)
        xl = atleast_1d(lower_bounds_concrete_in)

        if tangent_point is None:
            tangent_point = (xu + xl)/2  # Optimal tangent point

        act = self.forward(tangent_point)
        a = self.derivative(tangent_point)
        b = act - a * tangent_point

        return torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)

    # noinspection DuplicatedCode
    def _intercept_line(self,
                        lower_bounds_concrete_in: torch.Tensor,
                        upper_bounds_concrete_in: torch.Tensor) -> torch.Tensor:

        """
        Calculates the line intercepting two points of the activation function.

        Args:
            lower_bounds_concrete_in:
                The array with lower bounds for each neuron_num.
            upper_bounds_concrete_in:
                The array with upper bounds for each neuron_num.

        Returns:
            An array where the first column is a and the second column is b, the
            parameters in l(x) = ax + b.
        """

        xu = atleast_1d(upper_bounds_concrete_in)
        xl = atleast_1d(lower_bounds_concrete_in)

        a = (self.forward(xu) - self.forward(xl)) / (xu - xl)
        b = self.forward(xu) - a * xu

        return torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)

    def _iterative_minimal_tangent_line(self,
                                        xl: torch.Tensor,
                                        xu: torch.Tensor,
                                        upper: bool = True) -> torch.Tensor:

        """
        Uses the iterative tangent method described in the paper to find the valid
        tangent line coordinate closest to 0.

        Args:
            xl:
                The array with lower bounds for each neuron_num.
            xu:
                The array with upper bounds for each neuron_num.
            upper:
                If true an upper bound is calculated, else a lower bound is calculated.

        Returns:
            An array where the first column is a and the second column is b, the
            parameters in l(x) = ax + b
        """

        if upper:
            x_bound = xl
            xi = xu
        else:
            x_bound = xu
            xi = xl

        for i in range(self._num_iter_min_tangent_line):
            xi = self._update_xi(xi, x_bound, upper)

        line = self._tangent_line(xl, xu, xi)
        return line

    def derivative(self, x: torch.Tensor) -> torch.Tensor:

        """
        Calculates the derivative of the activation function at input x.

        Args:
            x: The input
        Returns:
            The derivative of the activation function at x
        """

        raise NotImplementedError("derivative(...) not implemented in subclass")

    def _update_xi(self, xi: torch.Tensor, x_bound: torch.Tensor, upper: bool):

        """
        Calculates the new xi for the iterative tangent method.

        Args:
            xi:
                The last tangent point calculated.
            x_bound:
                The lower/upper input bound for calculating upper/lower relaxation
                respectively.
            upper:
                If true the upper tangent is calculated, else the lower tangent is
                calculated.
        """

        raise NotImplementedError("_update_xi(...) not implemented in subclass")

    def integral(self, xl: torch.Tensor, xu: torch.Tensor) -> torch.Tensor:

        """
        Calculates the integral of the activation function from xl to xu.

        Args:
            xl:
                The lower bound.
            xu:
                The upper bounds.
        Returns:
            The integral from xl to xu.
        """

        raise NotImplementedError("integral(...) not implemented in subclass")


class Sigmoid(AbstractSShaped):

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.modules.activation.Sigmoid, nn.Sigmoid]

    # noinspection PyTypeChecker
    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding neuron_num. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return 1 / (1 + torch.exp(-x))

    def ssip_forward(self, bounds_symbolic_pre: torch.Tensor, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower bounding equations of ssip.

        Args:
            bounds_symbolic_pre:
                A 2xNxM tensor with the symbolic bounds. The first dimension contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous neuron_num. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding neuron_num.
        """

        raise NotImplementedError(f"propagate_reversed(...) not implemented")

    def propagate_const_term(self, const_terms: torch.Tensor) -> torch.Tensor:

        """
        Propagates the constant part of the symbolic bound for ssip.

        Args:
            const_terms:
                A Nx2 tensor with the constant part of the lower equation in the first
                column and the upper in the second.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    # noinspection PyTypeChecker
    def derivative(self, x: torch.Tensor) -> torch.Tensor:

        """
        Calculates the derivative of the Sigmoid at input x.

        Args:
            x:
                The input.
        Returns:
            The derivative of the Sigmoid at x.
        """

        sigmoid = self.forward(x)
        return sigmoid * (1 - sigmoid)

    # noinspection PyTypeChecker
    def split_point(self, xl: torch.Tensor, xu: torch.Tensor) -> torch.Tensor:

        """
        Returns the preferred split point for the Sigmoid.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point.
        """

        mid = (self.forward(xu) + self.forward(xl)) / 2
        return -torch.log((1 / mid) - 1)

    def integral(self, xl: torch.Tensor, xu: torch.Tensor) -> torch.Tensor:

        """
        Returns the integral of the Sigmoid from xl to xu.

        Args:
            xl:
                The lower bound.
            xu:
                The upper bounds.
        Returns:
            The integral from xl to xu.
        """

        return torch.log(torch.exp(xu) + 1) - torch.log(torch.exp(xl) + 1)

    # noinspection PyTypeChecker
    def _update_xi(self, xi: torch.Tensor, x_bound: torch.Tensor, upper: bool) -> torch.Tensor:

        """
        Calculates the new xi for the iterative tangent method as described in the
        paper.

        Args:
            xi:
                The last tangent point calculated.
            x_bound:
                The lower/upper input bound for calculating upper/lower relaxation
                respectively.
            upper:
                If true the upper tangent is calculated, else the lower tangent is
                 calculated.
        """

        inner = 1 - 4 * (self.forward(xi) - self.forward(x_bound)) / (xi - x_bound)
        root = torch.sqrt(inner) / 2.

        if upper:
            sxi = 0.5 + root
        else:
            sxi = 0.5 - root
        new_xi = -torch.log(1 / sxi - 1)

        non_valid = torch.isnan(new_xi) + torch.isinf(new_xi)
        new_xi[non_valid] = xi[non_valid]  # Rounding error, use last valid relaxation.

        return new_xi


class Tanh(AbstractSShaped):

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.modules.activation.Tanh, nn.Tanh]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding neuron_num. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return torch.tanh(x)

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous neuron_num. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding neuron_num.
        """

        raise NotImplementedError(f"propagate_reversed(...) not implemented")

    def ssip_forward(self, bounds_symbolic_pre: torch.Tensor, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower bounding equations of ssip.

        Args:
            bounds_symbolic_pre:
                A 2xNxM tensor with the symbolic bounds. The first dimension contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    def propagate_const_term(self, const_terms: torch.Tensor) -> torch.Tensor:

        """
        Propagates the constant part of the symbolic bound for ssip.

        Args:
            const_terms:
                A Nx2 tensor with the constant part of the lower equation in the first
                column and the upper in the second.
        Returns:
            The post-op values
        """

        raise NotImplementedError(f"propagate(...) not implemented")

    # noinspection PyTypeChecker
    def derivative(self, x: torch.Tensor) -> torch.Tensor:

        """
        Calculates the derivative of the Tanh at input x.

        Args:
            x:
                The input.
        Returns:
            The derivative of the Tanh at x.
        """

        return 1 - self.forward(x) ** 2

    # noinspection PyTypeChecker,PyTypeChecker
    def split_point(self, xl: float, xu: float) -> float:

        """
        Returns the preferred split point for the Tanh.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point.
        """

        mid = (self.forward(xu) + self.forward(xl)) / 2
        return 0.5 * torch.log((1 + mid) / (1 - mid))

    def integral(self, xl: torch.Tensor, xu: torch.Tensor) -> torch.Tensor:

        """
        Returns the integral of the Tanh from xl to xu.

        Args:
            xl:
                The lower bounds.
            xu:
                The upper bounds.
        Returns:
            The integral from xl to xu
        """

        return torch.log(0.5*(torch.exp(-xu) + torch.exp(xu))) - torch.log(0.5*(torch.exp(-xl) + torch.exp(xl)))

    # noinspection PyTypeChecker
    def _update_xi(self, xi: torch.Tensor, x_bound:  torch.Tensor, upper: bool):

        """
        Calculates the new xi for the iterative tangent method as described in the
        paper.

        Args:
            xi:
                The last tangent point calculated.
            x_bound:
                The lower/upper input bound for calculating upper/lower relaxation
                respectively.
            upper:
                If true the upper tangent is calculated, else the lower tangent is
                calculated.
        """

        inner = 1 - (self.forward(xi) - self.forward(x_bound)) / (xi - x_bound)
        root = torch.sqrt(inner)
        root[inner < 0] = xi[inner < 0]  # Rounding error, use last valid upper relaxation.

        if upper:
            sxi = root
        else:
            sxi = - root
        new_xi = 0.5 * torch.log((1 + sxi) / (1 - sxi))
        new_xi[torch.isnan(new_xi)] = xi[torch.isnan(new_xi)]  # Rounding error, use last valid relaxation.

        return new_xi
