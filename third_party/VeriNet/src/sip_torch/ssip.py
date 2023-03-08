
"""
This file contains a torch implementation of standard SIP

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional

import torch

from verinet.sip_torch.sip import SIP
from verinet.sip_torch.operations.piecewise_linear import Relu
from verinet.neural_networks.verinet_nn import VeriNetNN
from verinet.util.config import CONFIG


class SSIP(SIP):

    """
    Implements the standard symbolic interval propagation (SSIP)
    """

    def __init__(self,
                 model: VeriNetNN,
                 input_shape: torch.LongTensor,
                 optimise_computations: bool = True,
                 optimise_memory: bool = True):

        """
        Args:
            model:
                The VeriNetNN neural network as defined in
                ../neural_networks/verinet_nn.py
            input_shape:
                The shape of the input, (input_size,) for 1D input or
                (channels, height, width) for 2D.
            optimise_computations:
                Deprecated
            optimise_memory:
                If true, symbolic bounds are only buffered when necessary for later
                splits.
        """

        super(SSIP, self).__init__(model, input_shape, optimise_computations, optimise_memory)

    def get_bounds_concrete_pre(self, node_num: int) -> Optional[list]:

        """
        Returns the concrete pre-operations bounds for the given node.

        Returns:
            A list where each element is a Nx2 matrix with the lower bound in
            the first column and upper in the second. Note that nodes with more
            than one input produce one tensor of bounds for each input.
        """

        node_num = self.num_nodes + node_num if node_num < 0 else node_num

        pre_concrete = self._get_pre_concrete(node_num=node_num)

        if pre_concrete is None:
            return None
        else:
            return [torch.cat((bounds[0, :, 0:1], bounds[1, :, 1:2]), dim=1) for bounds in pre_concrete]

    def get_bounds_concrete_post(self, node_num: int) -> Optional[torch.Tensor]:

        """
        Returns the concrete post-operations bounds for the given node.
        """

        node_num = self.num_nodes + node_num if node_num < 0 else node_num

        post_concrete = self._get_post_concrete(node_num=node_num)

        if post_concrete is None:
            return None
        else:
            return torch.cat((post_concrete[0, :, 0:1], post_concrete[1, :, 1:2]), dim=1)

    def bounds_symbolic_pre(self, node: int):

        """
        Returns the symbolic pre-operations bounds for the given node.
        """

        node = self.num_nodes + node if node < 0 else node
        return node.bounds_symbolic_pre

    # noinspection PyArgumentList,PyTypeChecker
    def calc_bounds(self, bounds_concrete_in: torch.Tensor, from_node: int = None, to_node: int = None) -> bool:

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

        bounds_concrete_in = bounds_concrete_in.clone().to(device=self._device)
        bounds_concrete_in = torch.cat((bounds_concrete_in.unsqueeze(0), bounds_concrete_in.unsqueeze(0)), dim=0)
        bounds_concrete_in = self._adjust_bounds_from_forced([bounds_concrete_in], self.nodes[0].forced_bounds_pre)

        if self._tensor_type == torch.FloatTensor:
            self.nodes[0].bounds_concrete_pre = [bound.float() for bound in bounds_concrete_in]
            self.nodes[0].bounds_concrete_post = bounds_concrete_in[0].float()
        else:
            self.nodes[0].bounds_concrete_pre = [bound.double() for bound in bounds_concrete_in]
            self.nodes[0].bounds_concrete_post = bounds_concrete_in[0].double()

        from_node = 1 if (from_node is None or from_node < 1) else from_node
        to_node = self.num_nodes if (to_node is None or to_node > self.num_nodes) else to_node

        with torch.no_grad():
            for node_num in range(from_node, to_node):

                self.nodes[node_num].delete_bounds()

                success = self._prop_bounds(node_num)

                if node_num != to_node - 1:
                    self._clean_buffers(node_num)

                if not success:
                    return False

        return True

    def _prop_bounds(self, node_num: int) -> bool:

        """
        Calculates the symbolic input bounds.

        This updates all bounds and relaxations for the given node, by
        propagating from the previous node.

        Args:
            node_num:
                The node number
        Returns:
            True if the resulting concrete bounds are valid, else false.
        """

        node = self.nodes[node_num]

        node.bounds_symbolic_pre = self._get_pre_act_symb(node_num)

        if node.is_linear:
            node.bounds_symbolic_post = node.ssip_forward(node.bounds_symbolic_pre)

        else:
            if isinstance(node.op, Relu) and self._optimise_computations:
                node.bounds_concrete_pre = self._get_pre_concrete(node_num, lazy=True)
            else:
                node.bounds_concrete_pre = self._get_pre_concrete(node_num, lazy=False)

            relaxations = self._calc_relaxations(node_num)

            node.bounds_symbolic_post = self._prop_eq_trough_relaxation(*node.bounds_symbolic_pre, relaxations)

        if node.bounds_concrete_pre is None:
            return True
        else:
            for bounds in node.bounds_concrete_pre:
                if not self._valid_concrete_bounds(torch.cat((bounds[0, :, 0:1], bounds[1, :, 1:2]), dim=1)):
                    return False
            return True

    def _get_pre_act_symb(self, node_num: int) -> list:

        """
        Gets the input parameters for a node.

        Args:
            node_num:
                The node for which to calculate the post activation bounds
        Returns:
            A list of the symbolic_bounds_post  from connected nodes.
        """

        node = self.nodes[node_num]

        return [other_node.bounds_symbolic_post for other_node in self.nodes if other_node.idx in node.connections_from]

    def _get_pre_concrete(self, node_num: int, lazy: bool = False) -> Optional[list]:

        """
        Calculates the concrete pre-node bounds. If bounds are buffered in the node,
        those are returned instead.

        Args:
            node_num:
                The current node number
            lazy:
                If true, the operation is ReLU and the forced bounds are linear, the
                bounds are not recalculated.

        Returns:
            The concrete bounds for the node.
        """

        node = self.nodes[node_num]
        input_bounds = self.nodes[0].bounds_concrete_pre[0]

        if node.bounds_concrete_pre is not None:
            return node.bounds_concrete_pre

        if node.bounds_symbolic_pre is None:
            return None

        node.bounds_concrete_pre = []

        for i, bound in enumerate(node.bounds_symbolic_pre):
            if not lazy or (node.forced_bounds_pre is None) or (not isinstance(node.op, Relu)):
                bounds_low = self._calc_bounds_concrete(input_bounds[0], bound[0])
                bounds_up = self._calc_bounds_concrete(input_bounds[1], bound[1])

                bounds_concrete_pre = torch.cat((bounds_low.unsqueeze(0), bounds_up.unsqueeze(0)), dim=0)
                node.bounds_concrete_pre.append(bounds_concrete_pre)

            else:
                forced_bounds = node.forced_bounds_pre[i].clone().to(device=self._device)

                bounds_concrete_pre = torch.cat((forced_bounds.clone().unsqueeze(0),
                                                 forced_bounds.clone().unsqueeze(0)), dim=0)

                for j in range(2):

                    non_lin_idx = ((bounds_concrete_pre[j, :, 0] < 0) * (bounds_concrete_pre[j, :, 1] > 0))

                    bounds_concrete_pre[j][non_lin_idx] = self._calc_bounds_concrete(input_bounds[j],
                                                                                     bound[j, non_lin_idx])
                node.bounds_concrete_pre.append(bounds_concrete_pre)

        node.bounds_concrete_pre = self._adjust_bounds_from_forced(node.bounds_concrete_pre,
                                                                   node.forced_bounds_pre)

        return node.bounds_concrete_pre

    def _get_post_concrete(self, node_num: int) -> Optional[torch.Tensor]:

        """
        Calculates the concrete post-node bounds.

        If bounds are buffered in self._bounds_concrete_post, those are returned
        instead. It is assumed that the necessary symbolic equations/ concrete bounds
        for previous nodes are already calculated.

        Args:
            node_num:
                The current node number
        Returns:
            The concrete bounds for 'node'.
        """

        node = self.nodes[node_num]
        input_bounds = self.nodes[0].bounds_concrete_pre[0]

        if node.bounds_concrete_post is not None:
            return node.bounds_concrete_post

        node.bounds_concrete_post = torch.zeros((2, self.nodes[node_num].out_size, 2),
                                                dtype=self._precision).to(device=self._device)

        if node.bounds_concrete_pre is not None and node.op.is_monotonically_increasing:

            if len(node.bounds_concrete_pre) > 1:
                bounds_pre_low = torch.cat([bounds[0].unsqueeze(0) for bounds in node.bounds_concrete_pre], dim=0)
                bounds_pre_up = torch.cat([bounds[1].unsqueeze(0) for bounds in node.bounds_concrete_pre], dim=0)
            else:
                bounds_pre_low, bounds_pre_up = node.bounds_concrete_pre[0][0], node.bounds_concrete_pre[0][1]

            node.bounds_concrete_post[0] = node.forward(bounds_pre_low)
            node.bounds_concrete_post[1] = node.forward(bounds_pre_up)

        else:
            if node.bounds_symbolic_post is None:
                return None

            for i in range(2):
                node.bounds_concrete_post[i] = self._calc_bounds_concrete(input_bounds[i], node.bounds_symbolic_post[i])

        return node.bounds_concrete_post

    def _clean_buffers(self, current_node_num: int):

        """
        Clears buffered values that are not used in the rest of the SIP process.

        A nodes symbolic bounds are considered not needed when all the
        following conditions are satisfied:

        1) They're not the input node

        2) They're not needed in the computational graph after the current node.

        Args:
            Current node:
                The current node number.
        """

        if not self._optimise_memory:
            return

        for node_num in range(self.num_nodes):

            node = self.nodes[node_num]

            node.bounds_symbolic_pre = None
            node.relaxations = None

            if node_num == (self.num_nodes - 1) or len(node.connections_to) == 0:
                # Keep post for output constraints.
                continue

            if not CONFIG.STORE_SSIP_BOUNDS:
                keep_post_bounds = False
                for connection_to in node.connections_to:
                    if connection_to > current_node_num:
                        keep_post_bounds = True
                        break

                if not keep_post_bounds:
                    node.bounds_symbolic_post = None

    def _calc_bounds_concrete(self, input_bounds: torch.Tensor, symbolic_bounds: torch.Tensor) -> torch.Tensor:

        """
        Calculates the concrete from the symbolic bounds.

        The concrete bounds are calculated by maximising/ minimising the symbolic
        bounds for each neuron.

        Args:
            input_bounds:
                A Mx2 tensor with the input bounds, where M is the input dimension of
                the network.
            symbolic_bounds :
                A Nx(M+1) tensor with the symbolic bounds, where N is the number of
                neurons in the node and M is the input dimension of the network.
        Returns
            A Nx2 tensor with the concrete_bounds.
        """

        coeffs_pos = symbolic_bounds[:, :-1].clone()
        coeffs_pos[coeffs_pos < 0] = 0
        coeffs_neg = symbolic_bounds[:, :-1].clone()
        coeffs_neg[coeffs_neg > 0] = 0

        in_lower = torch.unsqueeze(input_bounds[:, 0], dim=0)
        in_upper = torch.unsqueeze(input_bounds[:, 1], dim=0)

        bounds_concrete = torch.zeros((symbolic_bounds.shape[0], 2),
                                      dtype=self._precision).to(device=input_bounds.device)
        bounds_concrete[:, 0] = (torch.sum(coeffs_pos * in_lower, dim=1) +
                                 torch.sum(coeffs_neg * in_upper, dim=1) +
                                 symbolic_bounds[:, -1])

        bounds_concrete[:, 1] = (torch.sum(coeffs_pos * in_upper, dim=1) +
                                 torch.sum(coeffs_neg * in_lower, dim=1) +
                                 symbolic_bounds[:, -1])

        return bounds_concrete

    def _calc_relaxations(self, node_num: int) -> torch.Tensor:

        """
        Calculates the linear relaxations for the given node and concrete bounds.

        Returns:
            A 2xNx2 tensor where the first dimension indicates the lower and upper
            relaxation, the second dimension are the neurons in the current node and the
            last dimension contains the parameters [a, b] in l(x) = ax + b.
        """

        node = self.nodes[node_num]

        if len(node.bounds_concrete_pre) > 1:
            raise ValueError("Tried calculating relaxations for neuron_num with more than 1 input")

        low_relax = node.calc_linear_relaxation(node.bounds_concrete_pre[0][0, :, 0],
                                                node.bounds_concrete_pre[0][0, :, 1],
                                                prefer_parallel=False)[0]
        up_relax = node.calc_linear_relaxation(node.bounds_concrete_pre[0][1, :, 0],
                                               node.bounds_concrete_pre[0][1, :, 1],
                                               prefer_parallel=False)[1]

        node.relaxations = torch.cat((torch.unsqueeze(low_relax, 0), torch.unsqueeze(up_relax, 0)), dim=0)

        return node.relaxations

    @staticmethod
    def _prop_eq_trough_relaxation(bounds_symbolic: torch.Tensor, relaxations: torch.Tensor) -> torch.Tensor:

        """
        Propagates the given symbolic equations through the linear relaxations.

        Args:
            bounds_symbolic:
                A 2xNx(M+1) tensor with the symbolic bounds, where N is the number of
                neurons in the node and M is the number of input
            relaxations:
                A 2xNx2 tensor where the first dimension indicates the lower and upper
                relaxation, the second dimension contains the neurons in the current
                node and the last dimension contains the parameters
                [a, b] in l(const_terms) = ax + b.
        Returns:
            A Nx(M+1) with the new symbolic bounds.
        """

        bounds_symbolic_new = bounds_symbolic.clone()
        bounds_symbolic_new[0] *= relaxations[0, :, 0:1]
        bounds_symbolic_new[1] *= relaxations[1, :, 0:1]

        bounds_symbolic_new[0, :, -1] += relaxations[0, :, 1]
        bounds_symbolic_new[1, :, -1] += relaxations[1, :, 1]

        return bounds_symbolic_new

    # noinspection PyUnresolvedReferences
    def _adjust_bounds_from_forced(self,
                                   bounds_concrete: Optional[list],
                                   forced_input_bounds: Optional[list]) -> Optional[list]:

        """
        Adjusts the concrete input bounds using the forced bounds.

        The method chooses the best bound from the stored concrete input bounds and the
        forced bounds as the new concrete input bound.

        Args:
            bounds_concrete:
                A list of 2xNx2 tensors with the concrete lower and upper bounds for each
                input neuron_num.
            forced_input_bounds:
                A list of 2xNx2 tensors with the concrete lower and upper bounds for each
                input neuron_num.
        Returns:
            A list of 2xNx2 tensors with the concrete lower and upper bounds for each
            input neuron_num adjusted for the forced bounds.
        """

        if forced_input_bounds is None:
            return bounds_concrete
        elif bounds_concrete is None:
            return [torch.cat((bounds.unsqueeze(0), bounds.unsqueeze(0)).to(self._device), dim=0) for
                    bounds in forced_input_bounds]

        bounds_concrete_new = []

        for i in range(len(bounds_concrete)):

            bounds_concrete_this = bounds_concrete[i]
            forced_lower = forced_input_bounds[i][:, 0].to(device=self._device)
            forced_upper = forced_input_bounds[i][:, 1].to(device=self._device)

            # Adjust lower bounds
            smaller_idx = bounds_concrete_this[0, :, 0] <= forced_lower
            bounds_concrete_this[0, smaller_idx, 0] = forced_lower[smaller_idx]
            smaller_idx = bounds_concrete_this[0, :, 1] <= forced_lower
            bounds_concrete_this[0, smaller_idx, 1] = forced_lower[smaller_idx]
            larger_idx = bounds_concrete_this[0, :, 1] >= forced_upper
            bounds_concrete_this[0, larger_idx, 1] = forced_upper[larger_idx]

            # Adjust upper bounds
            smaller_idx = bounds_concrete_this[1, :, 0] <= forced_lower
            bounds_concrete_this[1, smaller_idx, 0] = forced_lower[smaller_idx]
            larger_idx = bounds_concrete_this[1, :, 0] >= forced_upper
            bounds_concrete_this[1, larger_idx, 0] = forced_upper[larger_idx]
            larger_idx = bounds_concrete_this[1, :, 1] >= forced_upper
            bounds_concrete_this[1, larger_idx, 1] = forced_upper[larger_idx]

            bounds_concrete_new.append(bounds_concrete_this)

        return bounds_concrete_new

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

    # noinspection PyArgumentList
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

    # noinspection PyTypeChecker
    def _init_datastructure(self):

        """
        Initialises the data-structure.
        """

        super(SSIP, self)._init_datastructure()

        # Set the correct symbolic equations for input node
        diagonal_idx = torch.arange(self.nodes[0].out_size)
        self.nodes[0].bounds_symbolic_post = torch.zeros((2, self.nodes[0].out_size, self.nodes[0].out_size + 1),
                                                         dtype=self._precision).to(device=self._device)
        self.nodes[0].bounds_symbolic_post[:, diagonal_idx, diagonal_idx] = 1
