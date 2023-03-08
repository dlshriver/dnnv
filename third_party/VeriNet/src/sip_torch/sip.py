"""
Abstract class for SIP-based algorithms

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional

import torch
import torch.nn as nn

from verinet.sip_torch.operations.abstract_operation import AbstractOperation
from verinet.sip_torch.operations.linear import Identity
from verinet.neural_networks.verinet_nn import VeriNetNN
from verinet.util.config import CONFIG
from verinet.util.logger import get_logger

logger = get_logger(CONFIG.LOGS_LEVEL_VERIFIER, __name__, "../../logs/", "sip_log")


class SIP:

    # noinspection PyTypeChecker
    def __init__(self,
                 model: VeriNetNN,
                 input_shape: torch.Tensor,
                 optimise_computations: bool = True,
                 optimise_memory: bool = True
                 ):

        """
        Args:
            model:
                The VeriNetNN neural network as defined in
                ../neural_networks/verinet_nn.py
            input_shape:
                The shape of the input, (input_size,) for 1D input or
                (channels, height, width) for 2D.
            optimise_computations:
                If true, only computations necessary to calculate the symbolic bounds
                at the output node are performed. Calculations of concrete bounds are
                only performed when necessary.
            optimise_memory:
                If true, symbolic bounds are only buffered when necessary for later
                splits.
        """

        if not isinstance(input_shape, torch.LongTensor):
            raise TypeError("Input shape should be a torch Long tensor")

        self._model = model
        self._device = model.device

        self._input_shape = input_shape
        self._optimise_computations = optimise_computations
        self._optimise_memory = optimise_memory

        self._precision = torch.float32 if CONFIG.PRECISION == 32 else torch.float64
        self._tensor_type = torch.FloatTensor if CONFIG.PRECISION == 32 else torch.DoubleTensor

        self._nodes = None
        self._forced_input_bounds: Optional[list] = None

        self._read_nodes_from_torch_model(model)
        self._init_datastructure()

    @property
    def num_nodes(self):
        return len(self._nodes)

    @property
    def input_dim(self):
        return int(torch.prod(self._input_shape))

    @property
    def nodes(self):
        return self._nodes

    @property
    def num_non_linear_neurons(self):

        return sum([node.num_non_linear_neurons for node in self.nodes])

    @property
    def has_cex_optimisable_relaxations(self) -> bool:

        """
        Returns true if any node in the network has cex-optimisable relaxtions.
        """

        for node in self.nodes:
            if node.has_cex_optimisable_relaxations:
                return True
        return False

    def set_non_parallel_relaxations(self):

        """
        Sets the relaxations of all nodes to the non-parallel relaxations.
        """

        for node in self.nodes:
            if node.relaxations_non_parallel is not None:
                node.relaxations = node.relaxations_non_parallel

    def set_parallel_relaxations(self):

        """
        Sets the relaxations of all nodes to the parallel relaxations.
        """

        for node in self.nodes:
            if node.relaxations_parallel is not None:
                node.relaxations = node.relaxations_parallel

    def set_optimised_relaxations(self):

        """
        Sets the relaxations of all nodes to the optimised relaxations.
        """

        for node in self.nodes:
            if node.relaxations_optimised is not None:
                node.relaxations = node.relaxations_optimised

    def get_forced_bounds_pre(self, copy: bool = False) -> list:

        """
        Returns a copied list of all nodes forced input bounds.

        Args:
            copy:
                If True, the bounds are deep-copied.
        """

        if not copy:
            return [node.forced_bounds_pre for node in self.nodes]

        else:
            forced_bounds = []
            for node in self.nodes:
                if node.forced_bounds_pre is None:
                    forced_bounds.append(None)
                else:
                    forced_bounds.append([bound.clone() for bound in node.forced_bounds_pre])

            return forced_bounds

    def set_forced_bounds_pre(self, forced_bounds: list):

        """
        Sets the forced input bounds of each node to the corresponding element in the
        list.
        """

        for i, node in enumerate(self._nodes):
            if forced_bounds[i] is not None:
                node.forced_bounds_pre = [bound.clone() for bound in forced_bounds[i]]

    def get_bounds_concrete_pre(self, node_num: int):

        """
        Returns a list with the concrete pre activation bounds for each neuron.
        """

        if self.nodes[node_num].bounds_concrete_pre is not None:
            return self.nodes[node_num].bounds_concrete_pre
        else:
            raise NotImplementedError()

    def get_bounds_concrete_post(self, node_num: int):

        """
        Returns a list with the concrete post activation bounds for each neuron.
        """

        if self.nodes[node_num].bounds_concrete_post is not None:
            return self.nodes[node_num].bounds_concrete_post
        else:
            raise NotImplementedError()

    def calc_bounds(self, bounds_concrete_in: torch.Tensor, from_node: int = 1, to_node: int = None) -> bool:

        """
        Calculate the bounds for all nodes in the network starting at from_node.

        Args:
            bounds_concrete_in:
                The constraints on the input. The first dimensions should be the same
                as the input to the neural network, the last dimension should contain
                the lower bound on axis 0 and the upper on axis 1.
            from_node:
                Updates from this node.
            to_node:
                Updates up to this node. If None, all remaining nodes are calculated.

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds
            are invalid if the forced bounds make at least one upper bound smaller than
            a lower bound.
        """

        raise NotImplementedError()

    def reset_datastruct(self):

        """
        Resets the symbolic datastructure
        """

        self._init_datastructure()

    def _init_datastructure(self):

        """
        Initialises the necessary datastructure
        """

        pass

    def convert_output_bounding_equation(self, output_equations: torch.Tensor,
                                         lower: bool = False,
                                         bias_sep_constraints: bool = False) -> Optional[torch.Tensor]:

        """
        Converts an equation wrt. output-variables to a lower/upper bounding equation
        with respect to the input-variables.

        Args:
            output_equations:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
            bias_sep_constraints:
                If true, the bias values from relaxation are calculated as separate
                values.
        Returns:
                [a_0, a_1 ... a_n,cb] in the resulting equation sum(a_i*x_i) + c where
                x_i are the networks input variables.

                If bias_sep_constraints is true, the coeffs
                [a_0, a_1 ... a_n, b_0, b_1 ... b_n, c] are calculated instead where
                a_i are the coeffs we get when always propagating through lower
                relaxations, while b_i indicate the effect change in equation
                by propagating through the upper relaxation of the i'th non-linear
                node instead of lower.
        """

        raise NotImplementedError()

    def _read_nodes_from_torch_model(self, torch_model: VeriNetNN):

        """
        Initializes the operations from the torch _model.

        Args:
            torch_model:
                The Neural Network
        """

        # Initialise with None for input node
        self._nodes = []

        for node in torch_model.nodes:
            self._nodes.append(self._process_node(node))

        self.nodes_sanity_check(self._nodes)

    @staticmethod
    def nodes_sanity_check(nodes):

        """
        Performs a simple sanity check on the network architecture.

        Args:
            nodes:
                The nodes in the network.
        """

        num_input_nodes = len([node for node in nodes if node.connections_from == []])
        num_output_nodes = len([node for node in nodes if node.connections_to == []])

        if num_input_nodes != 1:
            raise ValueError(f"VeriNet only supports networks with a single input layer (with connections_from = [])"
                             f"Got {num_input_nodes}")

        if num_output_nodes != 1:
            raise ValueError(f"VeriNet only supports networks with a single output layer (with connections_to = [])"
                             f"Got {num_output_nodes}")

        # if not isinstance(nodes[0].op, Identity) or not isinstance(nodes[-1].op, Identity):
        #     logger.warning("Expected a nn.Identity layer at the start and end of the network, behaviour without "
        #                    "may be undefined.")

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _process_node(self, verinet_nn_node: torch.nn):

        """
        Converts the given node from VeriNetNN node to SIPNode

        Args:
            verinet_nn_node:
                The torch node
        """

        if isinstance(verinet_nn_node, nn.Sequential):
            raise ValueError(f"nn.Sequential not implemented")

        # Get the operation
        try:
            op = AbstractOperation.get_activation_operation_dict()[verinet_nn_node.op.__class__]()
        except KeyError as e:
            raise ValueError(f"Operation in: {verinet_nn_node} not implemented") from e

        # Get the parameters of the operation
        for param in op.required_params:

            attr = getattr(verinet_nn_node.op, param)

            if (isinstance(attr, torch.FloatTensor) or isinstance(attr, torch.cuda.FloatTensor)) and \
                    self._tensor_type == torch.DoubleTensor:
                op.params[param] = attr.double()
            elif (isinstance(attr, torch.DoubleTensor) or isinstance(attr, torch.cuda.DoubleTensor)) and \
                    self._tensor_type == torch.FloatTensor:
                op.params[param] = attr.float()
            else:
                op.params[param] = attr

        if verinet_nn_node.idx == 0:
            in_shape = self._input_shape
        elif len(verinet_nn_node.connections_from) > 0:
            in_shape = self._nodes[verinet_nn_node.connections_from[0]].out_shape
        else:
            raise ValueError(f"Expected at least one connection to non-input node: {verinet_nn_node}")

        out_shape = op.out_shape(in_shape)
        op.params["in_shape"] = in_shape
        op.process_params()

        sip_node = SIPNode(verinet_nn_node.idx, op, verinet_nn_node.connections_from, verinet_nn_node.connections_to,
                           in_shape, out_shape)
        return sip_node

    def get_most_impactfull_neurons(self, output_weights: torch.Tensor = None) -> Optional[list]:

        """
        Returns a sorted list over the neurons heuristically determined to have the
        most impact on the weighted output.

        Args:
            output_weights:
                A Nx2 tensor with the weights for the lower bounds in column 1 and the
                upper bounds in column 2. All weights should be >= 0.
        Returns:
            List of tuples (node_num, neuron_num) sorted after which neuron are
            heuristically determined to have the most impact on the weighted output
        """

        raise NotImplementedError()

    def get_split_point(self, lower: float, upper: float, node_num: int) -> float:

        """
        Calculates the suggested split-point for a neuron.

        Args:
            lower:
                The lower input-bound to the neuron
            upper:
                The upper input-bound to the neuron
            node_num:
                The node number
        Returns:
            The split-point.
        """

        return self.nodes[node_num].get_split_point(lower, upper)

    def merge_current_bounds_into_forced(self):

        """
        Sets forced input bounds to the best of current forced bounds and calculated
        bounds.
        """

        forced_bounds = self.get_forced_bounds_pre()

        for i in range(self.num_nodes):

            bounds_concrete_pre = self.get_bounds_concrete_pre(i)

            if bounds_concrete_pre is None:
                continue

            elif forced_bounds[i] is None:
                forced_bounds[i] = [bounds.clone().cpu() for bounds in bounds_concrete_pre]

            else:
                for j in range(len(bounds_concrete_pre)):

                    bounds_concrete = bounds_concrete_pre[j].cpu()

                    better_lower = forced_bounds[i][j][:, 0] < bounds_concrete[:, 0]
                    forced_bounds[i][j][better_lower, 0] = bounds_concrete[better_lower, 0]

                    better_upper = forced_bounds[i][j][:, 1] > bounds_concrete[:, 1]
                    forced_bounds[i][j][better_upper, 1] = bounds_concrete[better_upper, 1]

        self.set_forced_bounds_pre(forced_bounds)

    @staticmethod
    def _adjust_bounds_from_forced(bounds_concrete: Optional[list], forced_input_bounds: Optional[list]) -> list:

        """
        Adjusts the concrete input bounds using the forced bounds.

        The method chooses the best bound from the stored concrete input bounds and the
        forced bounds as the new concrete input bound.

        Args:
            bounds_concrete:
                A list of 2xNx2 tensors with the concrete lower and upper bounds for each
                input node.
            forced_input_bounds:
                A list of 2xNx2 tensors with the concrete lower and upper bounds for each
                input node.
        Returns:
            A list of 2xNx2 tensors with the concrete lower and upper bounds for each
            input node adjusted for the forced bounds.
        """

        raise NotImplementedError()

    def get_neuron_bounding_equation(self,
                                     node_num: int,
                                     neuron_num: int,
                                     lower: bool = True,
                                     bias_sep_constraints: bool = False) -> list:

        """
        Returns the bounding equation for the given neuron.

        The true input of the neuron q(x) is constrained by the bounding equations:

        min_x q_l(x) < q(x) < max_x q_u(x)

        where q_l(x) is the lower bounding equation returned if lower=True, and
        q_u(x) is the upper.

        Args:
            node_num:
                The node number.
            neuron_num:
                The neuron number.
            lower:
                If true, the lower bounding equation is returned, else the upper.
            bias_sep_constraints:
                If true, the bias values from relaxation are calculated as separate
                values.
        Returns:
                [a_0, a_1 ... a_n,cb] in the resulting equation sum(a_i*x_i) + c where
                x_i are the networks input variables.

                If bias_sep_constraints is true, the coeffs
                [a_0, a_1 ... a_n, b_0, b_1 ... b_n, c] are calculated instead where
                a_i are the coeffs we get when always propagating through lower
                relaxations, while b_i indicate the effect change in equation
                by propagating through the upper relaxation of the i'th non-linear
                node instead of lower.
        """

        raise NotImplementedError()

    # noinspection PyTypeChecker
    @staticmethod
    def _valid_concrete_bounds(bounds_concrete: torch.Tensor) -> bool:

        """
        Checks that all lower bounds are smaller than their respective upper bounds.

        Args:
            bounds_concrete:
                A Nx2 tensor with the concrete lower and upper input bounds.
        Returns:
            True if the bounds are valid.
        """

        if bounds_concrete is None:
            return True

        else:
            # Check whether lower bounds are larger than the upper bounds
            idx = torch.nonzero(bounds_concrete[:, 1] < (bounds_concrete[:, 0]))

            if len(idx > 0) and ((bounds_concrete[idx, 1] + 1e-6) < bounds_concrete[idx, 0]).sum() > 0:
                # Invalid bounds; the current split constraints are invalid, thus the branch is safe.
                return False

            elif len(idx > 0):
                # Rounding error, continue with worst-case scenario.
                bounds_concrete[idx, 1], bounds_concrete[idx, 0] = bounds_concrete[idx, 0], bounds_concrete[idx, 1]
                return True

            else:
                return True


class SIPNode:

    """
    This class stores a sip operation as well as the connected nodes and computed
    values.
    """

    def __init__(self, idx: int, op: AbstractOperation, connections_from: list = None, connections_to: list = None,
                 in_shape: torch.tensor = None, out_shape: torch.Tensor = None):
        """
        Args:
            idx:
                The index of the node
            op:
                The torch.nn operation.
            connections_from:
                The indices of nodes that are connected to this node.
            connections_to:
                The indices of nodes this node is connected to.
            in_shape:
                The shape of the node's input.
            out_shape:
                The shape of the node's output.
        """

        self.idx = idx
        self.op = op

        self.connections_to = connections_to if connections_to is not None else []
        self.connections_from = connections_from if connections_from is not None else []

        self.relaxations = None
        self.relaxations_parallel = None
        self.relaxations_non_parallel = None
        self.relaxations_optimised = None

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.bounds_concrete_pre = None
        self.bounds_concrete_post = None
        self.bounds_symbolic_pre = None
        self.bounds_symbolic_post = None
        self.forced_bounds_pre = None

        # Values used for splitting heuristic
        self.intermediate_bounds = {}
        self.impact = None
        self.non_lin_indices = None

        # Relaxation biases.
        self.biases = {}
        self.biases_sum = {}
        self.relax_diff = {}

    @property
    def in_size(self) -> int:
        return int(torch.prod(self.in_shape))

    @property
    def out_size(self) -> int:
        return int(torch.prod(self.out_shape))

    @property
    def is_linear(self) -> bool:
        return self.op.is_linear

    @property
    def num_non_linear_neurons(self) -> int:

        """
        Returns the number of non-linear neurons based on the pre-operation bounds.
        """

        if self.is_linear:
            return 0

        if self.bounds_concrete_pre is None or len(self.bounds_concrete_pre) > 1:
            raise ValueError("Expected one set of bounds_concrete_pre")

        return self.op.get_num_non_linear_neurons(self.bounds_concrete_pre[0])

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        return self.op.forward(x, add_bias)

    def ssip_forward(self, symbolic_equations: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        return self.op.ssip_forward(symbolic_equations, add_bias)

    def rsip_backward(self, symbolic_equations: torch.Tensor, add_bias: bool = True):
        return self.op.rsip_backward(symbolic_equations, add_bias)

    def get_split_point(self, xl: float, xu: float) -> float:
        return self.op.split_point(xl, xu)

    def calc_linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                               prefer_parallel: bool = True) -> torch.Tensor:
        return self.op.linear_relaxation(lower_bounds_concrete_in, upper_bounds_concrete_in, prefer_parallel)

    def calc_out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:
        return self.op.out_shape(in_shape)

    def calc_optimised_relaxations(self, values):

        """
        Calculates the optimised relaxations for a specific set of values as
        calculated by the neural network.

        Args:
            values:
                The values of the node as calculated by the neural network
        Returns:
            The optimised relaxations.
        """

        if self.relaxations_non_parallel is None:
            return None
        else:
            self.relaxations_optimised = self.op.optimise_linear_relaxation(self.relaxations_non_parallel.clone(),
                                                                            self.bounds_concrete_pre,
                                                                            values)

        return self.relaxations_optimised

    def delete_bounds(self):

        self.bounds_concrete_pre = None
        self.bounds_concrete_post = None
        self.bounds_symbolic_pre = None
        self.bounds_symbolic_post = None

    @property
    def has_cex_optimisable_relaxations(self) -> bool:

        """
        Returns true the op node has cex-optimisable relaxtions.
        """

        return self.op.has_cex_optimisable_relaxations

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

        return self.op.get_non_linear_neurons(bounds_concrete_pre)

    def backprop_through_relaxation(self,
                                    bounds_symbolic_post: torch.Tensor,
                                    relaxations: torch.Tensor,
                                    lower: bool = True,
                                    get_relax_diff: bool = False):

        """
        Back-propagates through the node with relaxations

        Args:
            bounds_symbolic_post:
                The symbolic post activation bounds
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

        if len(self.connections_from) > 1:
            raise ValueError("Tried backprop through relaxation for node with more than one input")

        return self.op.backprop_through_relaxation(bounds_symbolic_post, self.bounds_concrete_pre[0], relaxations,
                                                   lower, get_relax_diff)

    def __call__(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        return self.forward(x, add_bias)

    def __str__(self):
        return f"VeriNetNNNode(idx: {self.idx}, op: {self.op}, " \
               f"to: {self.connections_to}, from: {self.connections_from}), " \
               f"in-shape: {self.in_shape}, out-shape: {self.out_shape}"

    def __repr__(self):
        return self.__str__()
