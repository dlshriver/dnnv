
"""
This file contains the Reversed Symbolic Interval Propagation (RSIP)

RSIP Calculates linear bounds on the networks output nodes, given bounds on the
networks input The current implementation supports box-constraints on the input.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from verinet.sip_torch.sip import SIP
from verinet.neural_networks.verinet_nn import VeriNetNN
from verinet.util.config import CONFIG


class RSIP(SIP):

    """
    Implements the reversed symbolic interval propagation (RSIP)
    """

    def __init__(self,
                 model: VeriNetNN,
                 input_shape: torch.LongTensor,
                 optimise_computations: bool = True,
                 optimise_memory: bool = True,
                 max_est_memory_usage: int = 12*10**9,
                 use_pbar: bool = False):

        """
        Args:
            model:
                The VeriNetNN neural network as defined in
                src/deep_split/neural_networks/deep_split_nn.py
            input_shape:
                The shape of the input, (input_size,) for 1D input or
                (channels, height, width) for 2D.
            optimise_computations:
                If true, only computations necessary to calculate the symbolic bounds
                at the output node are performed. Calculations of concrete bounds are
                only performed when necessary.
            optimise_memory:
                Disregarded, equations are never stored in RSIP.
            max_est_memory_usage:
                The maximum estimated memory usage accepted during computations in
                bytes.
            use_pbar:
                Determines whether to use progress bars.
        """

        self._max_estimated_memory_usage = max_est_memory_usage / 20

        # Determines when to recalculate all nodes instead of only unstable nodes.
        self.max_non_linear_rate = 0.75
        self.max_non_linear_rate_split_nodes = 0.25

        self._store_intermediate_bounds = CONFIG.HIDDEN_NODE_SPLIT
        self._calc_parallel_relaxations = CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS
        self._store_bias_values = False

        self._use_pbar = use_pbar

        super(RSIP, self).__init__(model, input_shape, optimise_computations, optimise_memory)

    def get_bounds_concrete_pre(self, node_num: int):

        """
        Returns a list with the concrete pre activation bounds for each node.
        """

        return self.nodes[node_num].bounds_concrete_pre

    # noinspection PyTypeChecker
    def get_bounds_concrete_post(self, node_num: int, force_recalculate: bool = False):

        """
        Returns a list with the concrete post activation bounds for each node.

        Args:
            node_num:
                The number of the node for which to get concrete bounds.
            force_recalculate:
                If False, buffered bounds may be used; these bounds are sound,
                but may be non-optimal.
        """

        node_num = self.num_nodes + node_num if node_num < 0 else node_num
        node = self.nodes[node_num]

        if not force_recalculate and node.bounds_concrete_post is not None:
            return node.bounds_concrete_post
        else:
            node.bounds_concrete_post = self._calc_concrete_bounds_post(node_num)

        return node.bounds_concrete_post

    # noinspection PyTypeChecker
    def update_modified_neurons(self, node_num: int):

        """
        Updates neurons at node that have changed from linear to non-linear or
        vice-versa.

        Checks the forced bounds of the given node to see if any changes has
        happened compared to the concrete_bounds_pre to determine whether a node
        has changed its linearity status. If so, the bounds and intermediate
        values of the node are recalculated.

        Args:
            node_num:
                The number of the node.
        """

        node = self.nodes[node_num]

        if node.is_linear:
            raise ValueError(f"Method called on node with linear op: {node}")
        if node.forced_bounds_pre is None:
            raise ValueError(f"Method called on node without forced bounds: {node}")
        if len(node.bounds_concrete_pre) != 1:
            raise ValueError(f"Expected 1 set of concrete bounds, got: {len(node.bounds_concrete_pre)}")

        node.delete_bounds()
        node.bounds_concrete_pre = [bounds.clone().to(device=self._device) for bounds in node.forced_bounds_pre]

        self._update_non_lin_indices(node_num)
        non_lin_neurons = torch.nonzero(node.get_non_linear_neurons(node.bounds_concrete_pre[0]))[:, 0].cpu()

        if self._store_intermediate_bounds:

            recalc_neurons = []

            for non_lin_neuron in non_lin_neurons:
                if non_lin_neuron not in node.non_lin_indices:
                    recalc_neurons.append(non_lin_neuron.reshape(1))

            if len(recalc_neurons) > 0:

                recalc_neurons = torch.cat(recalc_neurons)
                self._recalculate_neurons(node_num, recalc_neurons)

        node.relaxations = self._calc_relaxations(node_num)
        node.bounds_concrete_post = self._calc_weak_post_concrete_bounds(node_num)

    def _recalculate_neurons(self, node_num: int, neurons: torch.Tensor):

        """
        Recalculates the bounds, relaxation and intermediate bounds for the given
        neurons at the given node.

        Args:
            node_num:
                The number of the node
            neurons:
                A torch Long tensor with the indices of the neurons that should be
                recalculated.
        """

        node = self.nodes[node_num]

        node.non_lin_indices = torch.cat((node.non_lin_indices, neurons))

        mem_limited_indices = self._get_mem_limited_node_indices(node_num, len(neurons))

        self._calc_concrete_bounds_pre(node_num,
                                       optimise_computations=self._optimise_computations,
                                       non_linear_neurons=neurons,
                                       mem_limited_indices=mem_limited_indices)

        sorted_indices = torch.argsort(node.non_lin_indices, descending=False)

        for key1 in node.intermediate_bounds.keys():
            for key2 in node.intermediate_bounds[key1].keys():
                node.intermediate_bounds[key1][key2] = \
                    node.intermediate_bounds[key1][key2][sorted_indices]

        node.non_lin_indices = node.non_lin_indices[sorted_indices]

    # noinspection PyTypeChecker
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

        assert from_node >= 0, "From node should be >= 0"

        bounds_concrete_in = bounds_concrete_in.clone().to(device=self._device)

        if self._tensor_type == torch.DoubleTensor:
            bounds_concrete_in = bounds_concrete_in.double()
        elif self._tensor_type == torch.FloatTensor:
            bounds_concrete_in = bounds_concrete_in.float()

        bounds_concrete_in = self._adjust_bounds_from_forced([bounds_concrete_in], self.nodes[0].forced_bounds_pre)

        self.nodes[0].bounds_concrete_pre = bounds_concrete_in
        self.nodes[0].bounds_concrete_post = bounds_concrete_in[0]

        from_node = 1 if (from_node is None or from_node < 1) else from_node
        to_node = self.num_nodes if (to_node is None or to_node > self.num_nodes) else to_node

        pbar = self.get_pbar(inner=False, total_iters=to_node)
        pbar.update(from_node)

        with torch.no_grad():
            for node_num in range(from_node, to_node):

                pbar.update(1)

                node = self.nodes[node_num]
                node.delete_bounds()

                node.bounds_concrete_pre = self._calc_weak_pre_concrete_bounds(node_num)
                node.bounds_concrete_pre = self._adjust_bounds_from_forced(node.bounds_concrete_pre,
                                                                           node.forced_bounds_pre)
                node.intermediate_bounds = {}
                node.non_lin_indices = {}

                success = True
                if self._optimise_computations and self.nodes[node_num].is_linear:
                    node.bounds_concrete_post = self._calc_weak_post_concrete_bounds(node_num)
                else:
                    success = self._calc_concrete_bounds_pre(node_num, self._optimise_computations)
                if not success:
                    pbar.close()
                    return False

                if not node.is_linear:
                    node.relaxations = self._calc_relaxations(node_num)

                node.bounds_concrete_post = self._calc_weak_post_concrete_bounds(node_num)

        pbar.close()
        return True

    def _calc_concrete_bounds_pre(self, node_num: int, optimise_computations: bool = True,
                                  non_linear_neurons: torch.Tensor = None, mem_limited_indices: torch.Tensor = None):

        """
        Calculates the pre-operation concrete bounds for the given node.

        Args:
            node_num:
                The number of the node
            optimise_computations:
                If true, only computations necessary to calculate the symbolic bounds
                at the output node are performed. Calculations of concrete bounds are
                only performed when necessary.
            non_linear_neurons:
                A indexing tensor of neuron indices to be calculated, if None all
                non-linear neurons are calculated.
            mem_limited_indices:
                A list of [idx_0, idx_1..., idx_n] where idx_i to idx_i+1 are calculated
                at a time to fit everything into memory.
        Returns:
            True if the resulting concrete bounds are valid, else false.
        """

        node = self.nodes[node_num]
        input_bounds = self.nodes[0].bounds_concrete_pre[0]

        if non_linear_neurons is None or mem_limited_indices is None:
            non_linear_neurons, mem_limited_indices = self._get_non_linear_neurons(node_num, optimise_computations)

            if self._store_intermediate_bounds:
                node.non_lin_indices = non_linear_neurons.cpu() if non_linear_neurons is not None else None

        bounds_symbolic_pre = None

        pbar = self.get_pbar(inner=True, total_iters=len(mem_limited_indices))

        for i in range(len(mem_limited_indices) - 1):

            pbar.update(1)

            if non_linear_neurons is not None:
                indices = non_linear_neurons[mem_limited_indices[i]: mem_limited_indices[i + 1]]
            else:
                indices = torch.arange(mem_limited_indices[i], mem_limited_indices[i + 1])

            if bounds_symbolic_pre is None:
                bounds_symbolic_pre = [self._init_symb_bounds(node.in_size, indices)]
            else:
                bounds_symbolic_pre = [self._reset_symbolic_bounds(bounds_symbolic_pre[0], node.in_size, indices)]

            if bounds_symbolic_pre[0].shape[0] == 0:
                continue

            symb_low = self._backprop_symb_equations(node_num, bounds_symbolic_pre=bounds_symbolic_pre, lower=True)
            node.bounds_concrete_pre[0][indices, 0] = self._calc_concrete_bounds(input_bounds, symb_low[0], lower=True)

            del symb_low

            bounds_symbolic_pre = [self._reset_symbolic_bounds(bounds_symbolic_pre[0], node.in_size, indices)]
            symb_up = self._backprop_symb_equations(node_num, bounds_symbolic_pre=bounds_symbolic_pre, lower=False)
            node.bounds_concrete_pre[0][indices, 1] = self._calc_concrete_bounds(input_bounds, symb_up[0], lower=False)

        if self._store_intermediate_bounds:
            self._update_non_lin_indices(node_num)

        node.bounds_concrete_pre = self._adjust_bounds_from_forced(node.bounds_concrete_pre, node.forced_bounds_pre)

        pbar.close()

        for bounds in node.bounds_concrete_pre:
            if not self._valid_concrete_bounds(bounds):
                return False

        return True

    def _update_non_lin_indices(self, node_num):

        """
        Updates stored indices of non-linear bounds and corresponding
        intermediate_bounds in the given node num by removing all bounds corresponding
        to linear nodes.

        Args:
            node_num:
                The node index.
        """

        node = self.nodes[node_num]

        if len(node.bounds_concrete_pre) != 1:
            raise ValueError("Expected single input to node.")

        bounds_concrete_pre = node.bounds_concrete_pre[0][node.non_lin_indices]
        non_lin_this = torch.nonzero((node.get_non_linear_neurons(bounds_concrete_pre)))[:, 0]
        node.non_lin_indices = node.non_lin_indices[non_lin_this]

        for key1 in node.intermediate_bounds:
            for key2 in node.intermediate_bounds[key1]:
                node.intermediate_bounds[key1][key2] = node.intermediate_bounds[key1][key2][non_lin_this]

    # noinspection PyTypeChecker
    def _get_non_linear_neurons(self, node_num: int, optimise_computations: bool = True) -> tuple:

        """
        Returns the non-linear neurons of the node.

        Args:
            node_num:
                The node number
            optimise_computations:
                If true, only computations necessary to calculate the symbolic bounds
                at the output node are performed. Calculations of concrete bounds are
                only performed when necessary.

        Returns:
            A torch tensor with the indices of non-linear nodes as well as a tensor of
            indices where indices[i] to indices[i+1] of the index list are
            estimated to fit withing the memory as defined by the class parameters.
        """

        node = self.nodes[node_num]

        if len(node.connections_from) > 1:
            raise ValueError("Expected node to have one connection.")

        non_lin_nodes = None
        nodes_indices = []

        if not optimise_computations:
            nodes_indices = self._get_mem_limited_node_indices(node_num, node.in_size)

        elif optimise_computations:

            bounds_concrete_pre = node.bounds_concrete_pre[0]
            non_lin_nodes = torch.nonzero(node.get_non_linear_neurons(bounds_concrete_pre))[:, 0]
            non_lin_rate = non_lin_nodes.shape[0]/node.in_size

            if non_lin_nodes.shape[0] > 0:
                if (non_lin_rate > self.max_non_linear_rate or
                        (self.is_split_node(node_num) and non_lin_rate > self.max_non_linear_rate_split_nodes)):
                    nodes_indices = self._get_mem_limited_node_indices(node_num, node.in_size)
                else:
                    nodes_indices = self._get_mem_limited_node_indices(node_num, len(non_lin_nodes))

        return non_lin_nodes, nodes_indices

    def _calc_concrete_bounds_post(self, node_num: int):

        """
        Calculates the post-operation concrete bounds for the given node.

        Args:
            node_num:
                The number of the node
        Returns:
            True if the resulting concrete bounds are valid, else false.
        """

        node = self.nodes[node_num]
        input_bounds = self.nodes[0].bounds_concrete_pre[0]

        if node_num == 0:
            return node.bounds_concrete_post

        nodes_indices = self._get_mem_limited_node_indices(node_num, node.out_size)
        node.bounds_concrete_post = torch.zeros((node.out_size, 2), dtype=self._precision).to(device=self._device)

        for i in range(len(nodes_indices) - 1):

            indices = torch.arange(nodes_indices[i], nodes_indices[i + 1])

            bounds_symbolic_post = self._init_symb_bounds(node.out_size, indices)
            symb_low = self._backprop_symb_equations(node_num, bounds_symbolic_post=bounds_symbolic_post, lower=True)
            node.bounds_concrete_post[indices, 0] = self._calc_concrete_bounds(input_bounds, symb_low[0], lower=True)

            del symb_low, bounds_symbolic_post

            bounds_symbolic_post = self._init_symb_bounds(node.out_size, indices)
            symb_up = self._backprop_symb_equations(node_num, bounds_symbolic_post=bounds_symbolic_post, lower=False)
            node.bounds_concrete_post[indices, 1] = self._calc_concrete_bounds(input_bounds, symb_up[0], lower=False)

            del symb_up, bounds_symbolic_post

        return node.bounds_concrete_post

    def _calc_weak_pre_concrete_bounds(self, node_num: int) -> list:

        """
        Calculates the concrete pre-operation bounds for a node.

        This method calculates 'weak' bounds from the concrete post-operation output
        bounds of connected nodes instead of re-calculating from symbolic bounds.

        Args:
            node_num:
                The node for which to calculate pre-operation bounds.
        Returns:
            A list of the concrete pre-operation bounds.
        """

        node = self.nodes[node_num]
        node.bounds_concrete_pre = [other_node.bounds_concrete_post.clone() for other_node in self.nodes if
                                    other_node.idx in node.connections_from]

        return node.bounds_concrete_pre

    def _calc_weak_post_concrete_bounds(self, node_num: int) -> torch.Tensor:

        """
        Calculates the concrete post-operation bound.

        This method calculates 'weak' bounds by propagating the concrete
        pre-operation bounds through the relaxation instead of re-calculating from
        symbolic bounds.

        Args:
            node_num:
                The number of the node
        """

        node = self.nodes[node_num]
        bounds_pre = node.bounds_concrete_pre

        if node.is_linear:
            bounds_post = node.ssip_forward([bounds.T.reshape(2, -1, 1) for bounds in bounds_pre]).T.reshape(-1, 2)

        else:

            if node.relaxations is None:
                raise ValueError("Tried forward without relaxation")
            if len(node.connections_from) > 1:
                raise ValueError("Tried forward through relaxation with more than one input connection")

            relaxations = node.relaxations

            bounds_post = torch.zeros((bounds_pre[0].shape[0], 2), dtype=self._precision).to(device=self._device)
            bounds_post[:, 0] = bounds_pre[0][:, 0] * relaxations[0, :, 0] + relaxations[0, :, 1]
            bounds_post[:, 1] = bounds_pre[0][:, 1] * relaxations[1, :, 0] + relaxations[1, :, 1]

        node.bounds_concrete_post = bounds_post

        return bounds_post

    def _get_mem_limited_node_indices(self, node_num: int, num_bounds: int):

        """
        Returns a list with the from and to indices that can be calculated at once
        while within the memory limit.

        Args:
            node_num:
                The number of the node
            num_bounds:
                The number of pre-operation neurons for which to calculate bounds.
        Returns:
            A list of indices, where index[i] to index[i+1] can be calculated within
            the given memory limit.
        """

        largest_node = max([node.in_size for node in self.nodes[:node_num + 1]])
        max_size = int(self._max_estimated_memory_usage / (largest_node * 4))

        if max_size >= num_bounds:
            return [0, num_bounds]

        node_indices = [0]
        node_indices += [max_size] * (num_bounds // max_size)

        if (num_bounds % max_size) != 0:
            node_indices += [num_bounds % max_size]

        return np.cumsum(node_indices)

    def _backprop_symb_equations(self,
                                 node_num: int,
                                 bounds_symbolic_post: torch.Tensor = None,
                                 bounds_symbolic_pre: list = None,
                                 lower: bool = True) -> list:

        """
        Calculates the symbolic equations until to the input node.

        This method iteratively substitutes the bound-variables for the variables in
        the previous node until the input-node is reached.

        Notice that starting symbolic equations can be specified. This is interpreted
        as modifying node_num of the neural network. For example:

        The standard equations for a node of size 2 are initialised to:

        [[1, 0, 0],
         [0, 1, 0]]

        Where the first columns indicate the variables and the last column is the
        constant term. Specifying an equation like:

        [[1, -1, 0],
         [0, 2, 0]]

        Has the interpretation that we want to compute node 1 minus node 2 in the first
        row and 2 times node 2 in the second row instead.

        Args:
            node_num:
                The node number
            bounds_symbolic_post:
                Starting post-activation bounds for the given node. If None, the
                pre-activation bounds are used instead.
            bounds_symbolic_pre:
                Starting pre-activation bounds for the given node. If bounds_symbolic_post
                is not None, this value is disregarded. If both bounds_symbolic_post and
                bounds_symbolic_pre are none the bounds are automatically defined assuming
                that this is the starting node for backprop.
            lower:
                Specifies whether to calculate the lower or upper bound.

        Returns:
            The symbolic bounds.
        """

        node = self.nodes[node_num]
        node.biases = {}
        node.biases_sum = {}
        node.relax_diff = {}

        if bounds_symbolic_post is not None:
            node.bounds_symbolic_post = bounds_symbolic_post

        elif bounds_symbolic_pre is not None:

            if len(bounds_symbolic_pre) != 1:
                raise ValueError("Expected bounds_symbolic_pre to be of length 1")

            num_inputs = len(node.connections_from)
            node.bounds_symbolic_pre = [bounds_symbolic_pre[0].clone() for _ in range(num_inputs)]

            for i in range(1, num_inputs):
                node.bounds_symbolic_pre[i][:, -1] = 0

        else:
            raise ValueError("bounds_symbolic_pre or bounds_symbolic_post should be provided")

        for current_node_num in range(node_num, -1, -1):

            this_node = self.nodes[current_node_num]

            if this_node.bounds_symbolic_post is not None and this_node.bounds_symbolic_post.shape[0] > 0:

                if (not this_node.is_linear or current_node_num == 0) and \
                        (self._store_intermediate_bounds and current_node_num != node_num):

                    key1 = current_node_num
                    key2 = "low" if lower else "up"

                    if key1 not in node.intermediate_bounds.keys():
                        node.intermediate_bounds[key1] = {key2: this_node.bounds_symbolic_post.clone().cpu()}
                    else:
                        if key2 not in node.intermediate_bounds[key1].keys():
                            node.intermediate_bounds[key1][key2] = this_node.bounds_symbolic_post.clone().cpu()
                        else:
                            node.intermediate_bounds[key1][key2] = \
                                torch.cat((node.intermediate_bounds[key1][key2],
                                           this_node.bounds_symbolic_post.clone().cpu()),
                                          dim=0)

                this_node.bounds_symbolic_pre, biases, biases_sum, relax_diff = \
                    self._backprop_through_node(current_node_num, lower=lower, get_biases=self._store_bias_values)

                if self._store_bias_values:
                    node.biases[current_node_num] = biases
                    node.biases_sum[current_node_num] = biases_sum
                    node.relax_diff[current_node_num] = relax_diff

            if this_node.bounds_symbolic_pre is None:
                continue

            for i, connection in enumerate(this_node.connections_from):

                if self.nodes[connection].bounds_symbolic_post is None:
                    self.nodes[connection].bounds_symbolic_post = this_node.bounds_symbolic_pre[i]
                else:
                    self.nodes[connection].bounds_symbolic_post += this_node.bounds_symbolic_pre[i]

            if current_node_num != 0:
                this_node.bounds_symbolic_pre, this_node.bounds_symbolic_post = None, None

        bounds_symbolic_pre = self.nodes[0].bounds_symbolic_pre
        self.nodes[0].bounds_symbolic_pre, self.nodes[0].bounds_symbolic_post = None, None

        return bounds_symbolic_pre

    def _backprop_through_node(self, node_num: int, lower: bool, get_biases: bool = False) -> tuple:

        """
        Back-propagates the given symbolic bounds through one node.

        Args:
            node_num:
                The number of the node
            lower:
                Specifies whether to calculate the lower or upper bound.
            get_biases:
                If true, the biases from the relaxations are returned
                if false or non-relaxed node, None may be returned instead.

        Returns:
            The pre-operation symbolic bounds, biases, biases_sum and
            relax_diff.
        """

        node = self.nodes[node_num]

        biases, biases_sum, relax_diff = None, None, None

        if node.is_linear:
            bounds_symbolic_pre = node.rsip_backward(node.bounds_symbolic_post)
        else:
            assert node.relaxations is not None, "Expected relaxation to be pre-calculated"

            bounds_symbolic_pre, biases, biases_sum, relax_diff = \
                node.backprop_through_relaxation(node.bounds_symbolic_post, node.relaxations, lower=lower,
                                                 get_relax_diff=get_biases)

        return bounds_symbolic_pre, biases, biases_sum, relax_diff

    def _init_symb_bounds(self, node_size: int, indices: torch.Tensor):

        """
        Initialises and returns the symbolic bounds.

        The symbolic bounds are initialised as a tensor of dimensions
        (node_size, node_size + 1) with ones on the diagonal and zeros everywhere
        else. These bounds are intended for back-propagation.

        Args:
            node_size:
                The number of neurons in the node
            indices:
                Indices of the neurons for which to initialise symbolic bounds.

        Returns:
            The lower and upper symbolic bounds
        """

        bounds_symbolic = torch.zeros((len(indices), node_size + 1), dtype=self._precision).to(device=self._device)
        diag_idx = torch.arange(0, len(indices), dtype=torch.long).to(device=self._device)

        bounds_symbolic[diag_idx, indices] = 1

        return bounds_symbolic

    def _reset_symbolic_bounds(self, bounds_symbolic: torch.Tensor, node_size: int, indices: torch.Tensor):

        """
        Same as _init_symb_bounds using a pre-allocated tensor.

        Args:
            bounds_symbolic:
                The allocated tensor of shape (len(indices), node_size + 1)
            node_size:
                The number of neurons in the node
            indices:
                Indices of the neurons for which to initialize symbolic bounds.

        Returns:
            The lower and upper symbolic bounds

        """

        if bounds_symbolic.shape[0] != len(indices) or bounds_symbolic.shape[1] != node_size + 1:
            return self._init_symb_bounds(node_size, indices)

        bounds_symbolic[:] = 0
        diag_idx = torch.arange(0, len(indices), dtype=torch.long).to(device=self._device)

        bounds_symbolic[diag_idx, indices] = 1

        return bounds_symbolic

    def _calc_concrete_bounds(self,
                              input_bounds: torch.Tensor,
                              symb_bounds: torch.Tensor,
                              lower: bool) -> torch.Tensor:

        """
        Calculates concrete bounds.

        Args:
            input_bounds:
                A Mx2 tensor with the input bounds, where M is the input dimension of
                the network.
            symb_bounds:
                A NxM tensor with the symbolic bounds, where M is the input dimension of
                the network and N are the number of nodes in the given node.
            lower:
                If True, the lower bounds are calculated, otherwise the upper.
        Returns:
            The concrete bounds.
        """

        bounds_concrete = torch.zeros((symb_bounds.shape[0]), dtype=self._precision).to(device=input_bounds.device)
        in_lower, in_upper = torch.unsqueeze(input_bounds[:, 0], dim=0), torch.unsqueeze(input_bounds[:, 1], dim=0)

        if not lower:
            in_lower, in_upper = in_upper, in_lower

        coeffs_pos = symb_bounds[:, :-1].clone()
        coeffs_pos[coeffs_pos < 0] = 0

        bounds_concrete[:] = torch.sum(coeffs_pos * in_lower, dim=1)

        coeffs_neg = symb_bounds[:, :-1].clone()
        coeffs_neg[coeffs_neg > 0] = 0

        bounds_concrete[:] += torch.sum(coeffs_neg * in_upper, dim=1)
        bounds_concrete[:] += symb_bounds[:, -1]

        return bounds_concrete

    # noinspection PyCallingNonCallable,PyTypeChecker
    def _calc_relaxations(self, node_num: int) -> torch.Tensor:

        """
        Calculates the linear relaxations for the given neuron_num and concrete bounds.

        Args:
            node_num:
                The number of the node.
        Returns:
            A 2xNx2 tensor where the first dimension indicates the lower and upper
            relaxation, the second dimension are the neurons in the current neuron_num and the
            last dimension contains the parameters [a, b] in l(x) = ax + b.
        """

        node = self.nodes[node_num]

        if len(node.bounds_concrete_pre) > 1:
            raise ValueError(f"Tried to calculate relaxation for node with multiple sets of concrete bounds:\n {node}")

        relaxations = node.calc_linear_relaxation(node.bounds_concrete_pre[0][:, 0],
                                                  node.bounds_concrete_pre[0][:, 1],
                                                  prefer_parallel=False)

        node.relaxations, node.relaxations_non_parallel = relaxations, relaxations

        if self._calc_parallel_relaxations:
            node.relaxations_parallel = node.calc_linear_relaxation(node.bounds_concrete_pre[0][:, 0],
                                                                    node.bounds_concrete_pre[0][:, 1],
                                                                    prefer_parallel=True)

        return node.relaxations

    def _adjust_bounds_from_forced(self, bounds_concrete: Optional[list], forced_input_bounds: Optional[list]) -> list:

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

        if forced_input_bounds is None:
            return bounds_concrete
        elif bounds_concrete is None:
            return [bounds.to(device=self._device).clone() for bounds in forced_input_bounds]

        for i in range(len(bounds_concrete)):

            if forced_input_bounds[i] is None:
                pass
            elif bounds_concrete[i] is None:
                bounds_concrete[i] = forced_input_bounds[i].clone().to(device=self._device)
            else:
                concrete = bounds_concrete[i]
                forced_lower = forced_input_bounds[i][:, 0].clone().to(device=bounds_concrete[i].device)
                forced_upper = forced_input_bounds[i][:, 1].clone().to(device=bounds_concrete[i].device)

                concrete[:, 0][concrete[:, 0] < forced_lower] = forced_lower[concrete[:, 0] < forced_lower]
                concrete[:, 0][concrete[:, 0] > forced_upper] = forced_upper[concrete[:, 0] > forced_upper]

                concrete[:, 1][concrete[:, 1] < forced_lower] = forced_lower[concrete[:, 1] < forced_lower]
                concrete[:, 1][concrete[:, 1] > forced_upper] = forced_upper[concrete[:, 1] > forced_upper]

        return bounds_concrete

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

        if bias_sep_constraints:
            self._store_bias_values = True
            self.set_parallel_relaxations()

        self._store_intermediate_bounds = False

        bounds_symbolic_pre = [torch.zeros((1, self.nodes[node_num].in_size + 1),
                                           dtype=self._precision).to(device=self._device)]
        bounds_symbolic_pre[0][0, neuron_num] = 1

        with torch.no_grad():
            if node_num != 0:
                bounds_symbolic_pre = self._backprop_symb_equations(node_num,
                                                                    bounds_symbolic_pre=bounds_symbolic_pre,
                                                                    lower=lower)

        self._store_intermediate_bounds = CONFIG.HIDDEN_NODE_SPLIT

        if bias_sep_constraints:
            self.set_non_parallel_relaxations()
            self._store_bias_values = False
            return [self.separate_bias(node_num, bounds_symbolic_pre[0])]
        else:
            return bounds_symbolic_pre

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

        if len(output_equations.shape) != 2 or output_equations.shape[1] != self.nodes[-1].out_size:
            raise ValueError(f"Expected equations of shape Nx{self.nodes[-1].out_size}, got {output_equations.shape}.")

        self.nodes[-1].intermediate_bounds = {}
        self.nodes[-1].non_lin_indices = {}

        if bias_sep_constraints:
            self.set_parallel_relaxations()
            self._store_bias_values = True

        # Calculate the bounding equation
        with torch.no_grad():
            const_terms = torch.zeros((output_equations.shape[0], 1), dtype=self._precision)
            output_equations = torch.cat((output_equations, const_terms), dim=1).to(device=self._device)

            bounds_symbolic_post = self._backprop_symb_equations(self.num_nodes-1,
                                                                 bounds_symbolic_post=output_equations,
                                                                 lower=lower)[0]

        if bias_sep_constraints:
            self._store_bias_values = False
            self.set_non_parallel_relaxations()
            return self.separate_bias(self.num_nodes-1, bounds_symbolic_post)
        else:
            return bounds_symbolic_post.clone()

    def set_parallel_relaxations(self):

        """
        Sets all relaxations to parallel lower and upper relaxations.
        """

        for node in self.nodes:
            if not node.is_linear:
                node.relaxations = node.relaxations_parallel

    def set_non_parallel_relaxations(self):

        """
        Sets all relaxations to non-parallel lower and upper relaxations.
        """

        for node in self.nodes:
            if not node.is_linear:
                node.relaxations = node.relaxations_non_parallel

    def separate_bias(self, node_num: int, bounds_symbolic_post: torch.Tensor):

        """
        Extracts the bias values of the given equation.

        For the given node_num, this method extracts the bias values for the
        relaxations from bounds_symbolic_post and appends them to the end.

        Args:
            node_num:
                The node idx for which to extract bias values from
                bounds_symbolic_post.
            bounds_symbolic_post:
                The bounds_symbolic_post corresponding to the bias values stored
                in self.nodes[node_num].biases and biases_sum.
        Returns:
            The bounds_symbolic_post matrix where [:, :self.nodes[0].in_size] are
            the coefficients of the input variables [:, self.nodes[0].in_size:-1] are
            the relaxation biases and [:, -1] are the constant parts of the equation.
        """

        node = self.nodes[node_num]
        bounds_symbolic_post = bounds_symbolic_post.cpu()
        new_bounds_symbolic_post = [bounds_symbolic_post[:, :-1]]
        constants = bounds_symbolic_post[:, -1]

        for i in range(node_num):

            if not self.nodes[i].is_linear:
                new_bounds_symbolic_post.append(node.relax_diff[i])
                constants -= node.biases_sum[i]

        new_bounds_symbolic_post.append(constants.view(-1, 1))

        return torch.cat(new_bounds_symbolic_post, dim=1)

    # noinspection PyArgumentList,PyCallingNonCallable
    def get_most_impactfull_neurons(self, output_equation: torch.Tensor = None, lower: bool = True) -> tuple:

        """
        Returns a sorted list over the neurons heuristically determined to have the
        most impact on the weighted output.

        Args:
            output_equation:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
        Returns:
            A tuple (impact, indices) where impact contains the sorted list of
            estimated impacts per neuron and indices contains the corresponding
            [node_num, neuron_num] pairs.
        """

        if not CONFIG.HIDDEN_NODE_SPLIT and not CONFIG.INPUT_NODE_SPLIT:
            raise ValueError("Expected at least one of CONFIG.HIDDEN_NODE_SPLIT or "
                             "CONFIG.INPUT_NODE_SPLIT to be True.")

        if not CONFIG.HIDDEN_NODE_SPLIT:

            self._calc_input_node_simple_impact(output_equation, lower=lower)
            impact = self._nodes[0].impact
            sorted_idx = torch.argsort(impact, descending=True)
            indices = torch.cat((torch.zeros((self._nodes[0].in_size, 1), dtype=torch.long).to(device=impact.device),
                                 sorted_idx.unsqueeze(1)), dim=1)
            return impact[sorted_idx], indices

        else:

            self._calc_non_linear_node_direct_impact(output_equation, lower=lower)
            self._calc_non_linear_node_indirect_impact()

            impacts = []
            indices = []

            for node in self._nodes:
                if not node.is_linear:

                    impact = node.impact
                    neuron_indices = node.non_lin_indices.unsqueeze(1)

                    node_nums = torch.zeros(len(neuron_indices), dtype=torch.long).unsqueeze(1) + node.idx
                    indices.append(torch.cat((node_nums, neuron_indices), dim=1))
                    impacts.append(impact)

            if CONFIG.INPUT_NODE_SPLIT:

                self._calc_input_node_indirect_impact()
                indices.append(torch.cat((torch.zeros(self._nodes[0].in_size, dtype=torch.long).unsqueeze(1),
                                          torch.LongTensor(list(range(self._nodes[0].in_size))).unsqueeze(1)), dim=1))
                impacts.append(self._nodes[0].impact)

            impacts = torch.cat(impacts, dim=0)
            indices = torch.cat(indices, dim=0)

            sorted_idx = torch.argsort(impacts, descending=True)

            return impacts[sorted_idx], indices[sorted_idx]

    def _calc_input_node_simple_impact(self, output_equation: torch.Tensor, lower: bool = True):

        """
        Estimates the indirect impact of the input nodes on the output equation

        This is a simplified version, used when intermediate symbolic equations
        are not available for non-linear hidden nodes due to
        CONFIG.HIDDEN_NODE_SPLIT = False.
        """

        input_node = self.nodes[0]

        bounds_width_input = (input_node.bounds_concrete_pre[0][:, 1] - input_node.bounds_concrete_pre[0][:, 0])
        bounds_symbolic_output = self.convert_output_bounding_equation(output_equation.view(1, -1), lower=lower)
        impact = bounds_width_input/2 * abs(bounds_symbolic_output[:, :-1])
        input_node.impact = impact[0]

    def _calc_non_linear_node_direct_impact(self, output_equation: torch.Tensor, lower: bool = True):

        """
        Estimates the impact of the nodes non-linear nodes on the output equation

        The direct impact is stored in node.intermediate_bounds["impact"]

        Args:
            output_equation:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
        """

        self.convert_output_bounding_equation(output_equation.view(1, -1), lower=lower)
        output_node = self.nodes[-1]

        for node in self._nodes:
            if not node.is_linear:

                if len(node.connections_from) != 1:
                    raise ValueError("Expected one input connection.")

                idx = node.non_lin_indices

                if lower:
                    symb_bounds_in_neg = output_node.intermediate_bounds[node.idx]['low'][:, idx].clone()
                    symb_bounds_in_neg[symb_bounds_in_neg > 0] = 0
                    biases = symb_bounds_in_neg * node.relaxations[1, node.non_lin_indices, 1].cpu().view(1, -1)
                else:
                    symb_bounds_in_pos = output_node.intermediate_bounds[node.idx]['up'][:, idx].clone()
                    symb_bounds_in_pos[symb_bounds_in_pos < 0] = 0
                    biases = symb_bounds_in_pos * node.relaxations[1, node.non_lin_indices, 1].cpu().view(1, -1)

                node.impact = torch.sum(biases, dim=0) + 1e-6

    def _calc_non_linear_node_indirect_impact(self):

        """
        Estimates the indirect impact of the non-linear nodes on the output equation

        It is assumed that the direct impact is already calculated for all nodes. The
        indirect impact is added to node.intermediate_bounds["impact"].
        """

        node_num = -1

        for i in range(self.num_nodes-1, -1, -1):
            node = self.nodes[i]

            if not node.is_linear:

                node_num += 1

                if len(node.connections_from) != 1:
                    raise ValueError("Expected one input connection.")

                this_idx = node.non_lin_indices
                if this_idx.shape[0] == 0:
                    continue

                relaxations = node.relaxations.cpu()

                for succ_node in self.nodes[i+1:]:
                    if not succ_node.is_linear:

                        if len(succ_node.connections_from) != 1:
                            raise ValueError("Expected one input connection.")

                        succ_idx = succ_node.non_lin_indices
                        if succ_idx.shape[0] == 0:
                            continue

                        succ_intermediate_bounds = succ_node.intermediate_bounds[i]
                        succ_bounds = succ_node.bounds_concrete_pre[0][succ_idx]

                        symb_bounds_in_neg_low = succ_intermediate_bounds['low'][:, this_idx].clone()
                        symb_bounds_in_neg_low[symb_bounds_in_neg_low > 0] = 0

                        symb_bounds_in_pos_up = succ_intermediate_bounds['up'][:, this_idx].clone()
                        symb_bounds_in_pos_up[symb_bounds_in_pos_up < 0] = 0

                        biases_low = symb_bounds_in_neg_low * relaxations[1, this_idx, 1].view(1, -1)
                        biases_up = symb_bounds_in_pos_up * relaxations[1, this_idx, 1].view(1, -1)

                        relative_impact = torch.clip(torch.abs(biases_low / succ_bounds[:, 0].view(-1, 1).cpu()), 0, 1)
                        relative_impact += torch.clip(torch.abs(biases_up / succ_bounds[:, 1].view(-1, 1).cpu()), 0, 1)

                        indirect_impact = relative_impact * succ_node.impact.view(-1, 1)

                        node.impact += (indirect_impact.sum(dim=0) *
                                        CONFIG.INDIRECT_HIDDEN_MULTIPLIER)

    def _calc_input_node_indirect_impact(self):

        """
        Estimates the indirect impact of the input nodes on the output equation

        It is assumed that the direct impact is already calculated for all nodes. The
        indirect impact is stored in node.intermediate_bounds["impact"].
        """

        node = self.nodes[0]

        bounds_width_input = (node.bounds_concrete_pre[0][:, 1] - node.bounds_concrete_pre[0][:, 0]).cpu()
        node.impact = torch.zeros(node.in_size, dtype=self._precision)

        for j in range(1, self.num_nodes):

            other_node = self.nodes[j]

            if not other_node.is_linear:

                if len(other_node.connections_from) != 1:
                    raise ValueError("Expected one input connection.")

                idx = other_node.non_lin_indices
                if len(idx) == 0:
                    continue

                bounds_width = (other_node.bounds_concrete_pre[0][idx, 1] -
                                other_node.bounds_concrete_pre[0][idx, 0]).cpu()
                impact = bounds_width_input.view(1, -1)/2 * (abs(other_node.intermediate_bounds[0]["low"][:, :-1]) +
                                                             abs(other_node.intermediate_bounds[0]["up"][:, :-1]))
                relative_impact = impact/bounds_width.view(-1, 1)
                indirect_impact = torch.sum(other_node.impact.view(-1, 1) * relative_impact,
                                            dim=0)

                node.impact += indirect_impact * CONFIG.INDIRECT_INPUT_MULTIPLIER

    def is_split_node(self, node_num: int) -> bool:

        """
        Checks whether there is a recurrent connection in the computational graph
        between node_num and the next non-linear node.

        Args:
            node_num:
                The current node number.
        Returns:
            True if there is a recurrent connection in the computational graph
            between node_num and the first succeeding non-linear node.
        """

        node = self.nodes[node_num]

        while True:

            if len(node.connections_to) == 0:
                return False

            if len(node.connections_to) >= 2:
                return True

            elif not node.is_linear:
                return False

            else:
                node = self.nodes[node.connections_to[0]]

    def get_pbar(self, inner: bool, total_iters: int):

        """
        Returns the pbar object.

        Args:
            inner:
                If true, the inner loop (batch-processing) pbar is returned,
                otherwise the outer (node-processing) pbar.
            total_iters:
                The total number of iterations expected for the pbar.
        Returns:
            The pbar
        """

        desc = "RSIP processing batch" if inner else "RSIP processing node"
        if self._use_pbar:
            return tqdm(total=total_iters, leave=False, desc=desc)
        else:
            return NOPPBAR()


class NOPPBAR:

    """
    A dummy object for progress bar doing no operations.
    """

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
