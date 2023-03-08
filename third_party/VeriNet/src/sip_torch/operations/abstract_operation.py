
"""
This file contains abstractions for the network operations. Note that both layers
(FC, Conv, ...) and activation functions (ReLU, Sigmoid, ...) are considered network
operations.

The abstractions are used to calculate linear relaxations, function values, and
derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch

from verinet.util.config import CONFIG


class AbstractOperation:

    """
    Abstract class for describing network operations.

    This class should be subclassed to add support for layers (FC, Conv, ...) and
    activation functions (ReLU, Sigmoid, ...).

    The main purpose of this class is to provide a standardised implementation to
    calculate function values, derivatives and linear relaxations for all supported
    network operations.
    """

    def __init__(self):
        self.params = {}

        self._precision = torch.float32 if CONFIG.PRECISION == 32 else torch.float64
        self._tensor_type = torch.FloatTensor if CONFIG.PRECISION == 32 else torch.DoubleTensor

    @property
    def is_linear(self) -> bool:

        """
        Returns true if the operation is linear.
        """

        raise NotImplementedError(f"is_linear(...) not implemented")

    @property
    def is_monotonically_increasing(self) -> bool:
        raise NotImplementedError(f"is_linear(...) not implemented")

    @property
    def required_params(self) -> list:

        """
        Returns a list of the required parameters of the operation.
        """

        return []

    # noinspection PyUnresolvedReferences
    @classmethod
    def get_subclasses(cls):

        """
        Returns a generator yielding all subclasses of this class.

        This is intended to only be used in get_activation_mapping_dict().
        """

        # Subclasses in not-imported files are not found
        import verinet.sip_torch.operations.s_shaped
        import verinet.sip_torch.operations.piecewise_linear
        import verinet.sip_torch.operations.linear

        for subclass in cls.__subclasses__():
            # noinspection PyUnresolvedReferences
            yield from subclass.get_subclasses()
            yield subclass

    @staticmethod
    def get_activation_operation_dict():

        """
        Returns a dictionary mapping the torch activation functions and layers to the
        relevant subclasses.

        This is achieved by looping through all subclasses and calling
        abstracted_torch_funcs(), which simplifies adding new activation functions in
        the future.
        """

        activation_map = {}

        for subcls in AbstractOperation.get_subclasses():
            for torch_func in subcls.abstracted_torch_funcs():

                activation_map[torch_func] = subcls

        return activation_map

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Defines the torch activation functions abstracted by this class. For example,
        a ReLU abstraction might have: [nn.modules.activation.ReLU, nn.ReLU]

        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        raise NotImplementedError(f"abstracted_torch_funcs(...) not implemented")

    @property
    def has_cex_optimisable_relaxations(self) -> bool:

        """
        Returns true if the op-type can take advantage of cex-optimised relaxations.
        """

        return False

    def process_params(self):

        """
        Can be implemented in subclasses to do pre-processing on the args in params,
        is called from SIP immediately after adding all params.
        """

        pass

    def get_num_non_linear_neurons(self, bounds_concrete_pre: torch.Tensor) -> int:

        """
        Returns the number of non-linear neurons based on bounds.

        Args:
            bounds_concrete_pre:
                A Nx2 tensor with the lower bounds of the neurons in the first col
                and upper bounds in the second.
        """

        if self.is_linear:
            return 0
        else:
            raise NotImplementedError()

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

    def split_point(self, xl: float, xu: float) -> float:

        """
        Calculates the preferred split point for branching.

        Args:
            xl:
                The lower bound on the input.
            xu:
                The upper bound on the input.

        Returns:
            The preferred split point
        """

        return (xl + xu)/2

    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Calculates the linear relaxation

        The linear relaxation is a 2xNx2 tensor, where relaxation[0] represents a and b
        of the lower relaxation equation: l(const_terms) = ax + b (correspondingly,
        relaxation[1] is the upper relaxation).

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

        raise NotImplementedError(f"linear_relaxation(...) not implemented")

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

        return relaxation

    # noinspection PyTypeChecker
    def _linear_relaxation_equal_bounds(self,
                                        lower_bounds_concrete_in: torch.Tensor,
                                        upper_bounds_concrete_in: torch.Tensor,
                                        relaxations: torch.Tensor):

        """
        Calculates the linear relaxations for the nodes with equal lower and upper
        bounds.

        Added for convenience; this function is intended to be called from
        linear_relaxation in subclasses.

        The linear relaxations used when both bounds are equal is l=ax + b with a = 0
        and b equal the activation at the bounds. The relaxations array is modified in
        place and is assumed to be initialized to 0.

        Args:
            lower_bounds_concrete_in:
                The concrete lower bounds of the input to the nodes.
            upper_bounds_concrete_in:
                The concrete upper bounds of the input to the nodes.
            relaxations:
                A Nx2 array with the a,b parameters of the relaxation lines for each
                node.
        """

        equal_idx = torch.nonzero(lower_bounds_concrete_in == upper_bounds_concrete_in)
        relaxations[equal_idx, 1] = self.forward(lower_bounds_concrete_in[equal_idx])

    # noinspection PyMethodMayBeStatic,PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return torch.LongTensor(in_shape)

    # noinspection PyCallingNonCallable
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

        if self.is_linear:
            return torch.BoolTensor(bounds_concrete_pre.shape[0])
        else:
            return torch.BoolTensor(bounds_concrete_pre.shape[0]) + 1

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

        lin_idx = torch.nonzero(~self.get_non_linear_neurons(bounds_concrete_pre))[:, 0]
        non_lin_idx = torch.nonzero(self.get_non_linear_neurons(bounds_concrete_pre))[:, 0]

        symb_bounds_in_pos = bounds_symbolic_post[:, non_lin_idx].clone()
        symb_bounds_in_neg = symb_bounds_in_pos.clone()
        symb_bounds_in_pos[symb_bounds_in_pos < 0] = 0
        symb_bounds_in_neg[symb_bounds_in_pos > 0] = 0

        non_linear_relaxations_a = relaxations[:, non_lin_idx, 0]
        non_linear_relaxations_b = relaxations[:, non_lin_idx, 1]

        biases, biases_sum, relax_diff = None, None, None

        # Linear neurons, lower and upper relaxations are the same.
        bounds_symbolic_post[:, lin_idx] = bounds_symbolic_post[:, lin_idx] * relaxations[0, lin_idx, 0].view(1, -1)
        bounds_symbolic_post[:, -1] += (bounds_symbolic_post[:, lin_idx] * relaxations[0, lin_idx, 1]).sum(dim=1)

        if lower:
            biases = symb_bounds_in_neg * non_linear_relaxations_b[1].view(1, -1)
            biases += symb_bounds_in_pos * non_linear_relaxations_b[0].view(1, -1)
            biases_sum = torch.sum(biases, dim=1)

            if get_relax_diff:
                bias_diff = non_linear_relaxations_b[1] - non_linear_relaxations_b[0]
                relax_diff = symb_bounds_in_neg * bias_diff
                relax_diff += symb_bounds_in_pos * bias_diff

            new_bounds = symb_bounds_in_neg * non_linear_relaxations_a[1].view(1, -1)
            new_bounds += symb_bounds_in_pos * non_linear_relaxations_a[0].view(1, -1)

        else:
            biases = symb_bounds_in_pos * non_linear_relaxations_b[1].view(1, -1)
            biases += symb_bounds_in_neg * non_linear_relaxations_b[0].view(1, -1)
            biases_sum = torch.sum(biases, dim=1)

            if get_relax_diff:
                bias_diff = non_linear_relaxations_b[1] - non_linear_relaxations_b[0]
                relax_diff = symb_bounds_in_neg * bias_diff
                relax_diff += symb_bounds_in_pos * bias_diff

            new_bounds = symb_bounds_in_neg * non_linear_relaxations_a[0].view(1, -1)
            new_bounds += symb_bounds_in_pos * non_linear_relaxations_a[1].view(1, -1)

        bounds_symbolic_post[:, non_lin_idx] = new_bounds
        bounds_symbolic_post[:, -1] += biases_sum

        return [bounds_symbolic_post], biases.cpu(), biases_sum.cpu(), relax_diff


class ActivationFunctionAbstractionException(Exception):
    pass
