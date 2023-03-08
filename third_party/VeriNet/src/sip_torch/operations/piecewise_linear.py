"""
This file contains abstractions for piecewise linear activation functions (ReLU, Identity ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn

from verinet.sip_torch.operations.abstract_operation import AbstractOperation
from verinet.util.config import CONFIG


class Relu(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.modules.activation.ReLU, nn.ReLU]

    @property
    def has_cex_optimisable_relaxations(self) -> bool:

        """
        Returns true if the op-type can take advantage of cex-optimised relaxations.
        """

        return True

    def get_num_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> int:

        """
        Returns the number of non-linear neurons based on bounds.

        Args:
            bounds_concrete_pre:
                A Nx2 tensor with the lower bounds of the neurons in the first col
                and upper bounds in the second.
        """

        return int(torch.sum((bounds_concrete_pre[:, 0] < 0) * (bounds_concrete_pre[:, 1] > 0)))

    def forward(self, x: torch.Tensor, add_bias: bool = True):

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

        return (x > 0) * x

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

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Calculates the linear relaxation

        The linear relaxation is a 2xNx2 tensor, where relaxation[0] represents a and b
        of the lower relaxation equation: l(const_terms) = ax + b (correspondingly,
        relaxation[1] is the upper relaxation).

        The relaxations are described in detail in the DeepSplit paper.

        Args:
            lower_bounds_concrete_in:
                The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in:
                The concrete upper bounds of the input to the nodes
            force_parallel:
                If true, parallel relaxations are required.
        Returns:
            The relaxations as a Nx2 tensor
        """

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = torch.zeros((2, layer_size, 2), dtype=self._precision).to(device=lower_bounds_concrete_in.device)

        # Operating in the positive area
        fixed_upper_idx = torch.nonzero(lower_bounds_concrete_in >= 0)
        relaxations[:, fixed_upper_idx, 0] = 1
        relaxations[:, fixed_upper_idx, 1] = 0

        # Operating in the negative area
        fixed_lower_idx = torch.nonzero(upper_bounds_concrete_in <= 0)
        relaxations[:, fixed_lower_idx, :] = 0

        # Operating in the non-linear area
        mixed_idx = torch.nonzero((upper_bounds_concrete_in > 0)*(lower_bounds_concrete_in < 0))

        if len(mixed_idx) == 0:
            return relaxations

        xl = lower_bounds_concrete_in[mixed_idx]
        xu = upper_bounds_concrete_in[mixed_idx]

        # Upper relaxation
        a = xu / (xu - xl)
        b = - a * xl
        relaxations[1, :, 0][mixed_idx] = a
        relaxations[1, :, 1][mixed_idx] = b

        # Lower relaxation
        if force_parallel:
            relaxations[0, :, 0][mixed_idx] = a
        else:
            larger_up_idx = (upper_bounds_concrete_in + lower_bounds_concrete_in) > 0
            smaller_up_idx = ~larger_up_idx

            relaxations[0, larger_up_idx, 0] = 1
            relaxations[0, smaller_up_idx, 0] = 0

        return relaxations

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def optimise_linear_relaxation(self, relaxation: torch.Tensor,
                                   bounds: torch.Tensor,
                                   values: torch.Tensor) -> torch.Tensor:

        """
        Optimised the given linear relaxation with respect to a set of values as
        calculated by the neural network.

        Args:
            relaxation:
                The current linear relaxation.
            bounds:
                The concrete pre-activation bounds for the node.
            values:
                The values of the node as calculated by the neural network
        Returns:
            The relaxations as a Nx2 array
        """

        pos_idx = values > 0
        neg_idx = values <= 0

        relaxation[0, pos_idx, 0] = 1
        relaxation[0, neg_idx, 0] = 0
        relaxation[0, :, 1] = 0

        max_bounds_multiple = CONFIG.OPTIMISED_RELU_RELAXATION_MAX_BOUNDS_MULTIPLIER

        bounds_ratio = bounds[0][:, 1] / abs(bounds[0][:, 0])
        relaxation[0, bounds_ratio < 1/max_bounds_multiple, 0] = 0
        relaxation[0, bounds_ratio > max_bounds_multiple, 0] = 1

        return relaxation

    def split_point(self, xl: float, xu: float):

        """
        Returns the preferred split point for branching which is 0 for the ReLU.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point
        """

        return 0

    def get_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> torch.Tensor:

        """
        An array of boolean values. 'True' indicates that the corresponding neuron
        is non-linear in the input domain as described by bounds_concrete_pre; 'false'
        that it is linear.

        Args:
            bounds_concrete_pre:
                The concrete pre-operation value.
        Returns:
            A boolean tensor with 'true' for non-linear neurons and false otherwise.
        """

        return (bounds_concrete_pre[:, 0] < 0) * (bounds_concrete_pre[:, 1] > 0)

    # noinspection PyUnusedLocal
    def backprop_through_relaxation(self,
                                    bounds_symbolic_post: torch.Tensor,
                                    bounds_concrete_pre: torch.Tensor,
                                    relaxations: torch.Tensor,
                                    lower: bool = True,
                                    get_relax_diff: bool = False) -> tuple:

        """
        Back-propagates through the node with relaxations

        Args:
            bounds_symbolic_post:
                The symbolic post activation bounds
            bounds_concrete_pre:
                The concrete pre activation bounds
            relaxations:
                A 2xNx2 tensor where the first dimension indicates the lower and upper
                relaxation, the second dimension are the neurons in the current node
                and the last dimension contains the parameters [a, b] in l(x) = ax + b.

            lower:
                If true, the lower bounds are calculated, else the upper.
            get_relax_diff:
                If true, the differences between the lower and upper relaxations
                are returned. If false or non-relaxed node, None may be returned instead.

        Returns:
            The resulting symbolic bounds, biases, biases_sum and relax_diff. Biases
            are the constants added to the symbolic equation, relax_diff are the
            differences between the upper and lower relaxation.
        """

        lower_idx = bounds_concrete_pre[:, 1] <= 0
        non_lin_idx = torch.nonzero(self.get_non_linear_neurons(bounds_concrete_pre))[:, 0]

        symb_bounds_in_pos = bounds_symbolic_post[:, non_lin_idx]
        symb_bounds_in_neg = symb_bounds_in_pos.clone()
        symb_bounds_in_pos[symb_bounds_in_pos < 0] = 0
        symb_bounds_in_neg[symb_bounds_in_pos > 0] = 0

        bounds_symbolic_post[:, :-1][:, lower_idx] = 0

        non_linear_relaxations_a = relaxations[:, non_lin_idx, 0]
        non_linear_relaxations_b = relaxations[:, non_lin_idx, 1]

        biases, biases_sum, relax_diff = None, None, None

        if lower:
            biases = symb_bounds_in_neg * non_linear_relaxations_b[1].view(1, -1)
            biases_sum = torch.sum(biases, dim=1)

            if get_relax_diff:
                relax_diff = (symb_bounds_in_pos * non_linear_relaxations_b[1].view(1, -1)).cpu()
                relax_diff += (symb_bounds_in_neg * non_linear_relaxations_b[1].view(1, -1)).cpu()

            new_bounds = symb_bounds_in_neg * non_linear_relaxations_a[1].view(1, -1)
            new_bounds += symb_bounds_in_pos * non_linear_relaxations_a[0].view(1, -1)

        else:
            biases = symb_bounds_in_pos * non_linear_relaxations_b[1].view(1, -1)
            biases_sum = torch.sum(biases, dim=1)

            if get_relax_diff:
                relax_diff = (symb_bounds_in_pos * non_linear_relaxations_b[1].view(1, -1)).cpu()
                relax_diff += (symb_bounds_in_neg * non_linear_relaxations_b[1].view(1, -1)).cpu()

            new_bounds = symb_bounds_in_neg * non_linear_relaxations_a[0].view(1, -1)
            new_bounds += symb_bounds_in_pos * non_linear_relaxations_a[1].view(1, -1)

        bounds_symbolic_post[:, non_lin_idx] = new_bounds
        bounds_symbolic_post[:, -1] += biases_sum

        return [bounds_symbolic_post], biases.cpu(), biases_sum.cpu(), relax_diff
