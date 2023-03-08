
"""
Util classes for nn_bounds

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from enum import Enum

import numpy as np

from verinet.sip_torch.ssip import SIP
from verinet.verification.lp_solver import LPSolver
from verinet.util.config import CONFIG


class Status(Enum):
    """
    For keeping track of the verification_status.
    """

    Safe = 1
    Unsafe = 2
    Undecided = 3
    Underflow = 4


class Branch:
    """
    A class to keep track of the data needed when branching.
    """

    def __init__(self, depth: int, forced_bounds_pre: np.array, split_list: list):

        """
        Args:
            depth:
                The current split depth.
            forced_bounds_pre:
                The forced input bounds used in SIP.
            split_list:
                A list of dictionaries on the form:
                {"node": int, "neuron": int, "split_x": float, "upper": bool}
        """

        self._depth = depth

        self._forced_bounds_pre = forced_bounds_pre
        self._split_list = split_list

        self._lp_solver_constraints = None
        self.safe_constraints = []

    @property
    def depth(self):
        return self._depth

    @property
    def forced_bounds_pre(self):
        return self._forced_bounds_pre

    @forced_bounds_pre.setter
    def forced_bounds_pre(self, bounds):
        self._forced_bounds_pre = bounds

    @property
    def split_list(self):
        return self._split_list

    @property
    def lp_solver_constraints(self):
        return self._lp_solver_constraints

    @lp_solver_constraints.setter
    def lp_solver_constraints(self, constraints):
        self._lp_solver_constraints = constraints

    # noinspection PyTypeChecker
    @staticmethod
    def add_constraints_to_solver(sip: SIP, lp_solver: LPSolver, splits: list):

        """
        Creates the lp-solver constraint from the given split.

        Args:
            sip:
                The SIP object used to get the constraint equations.
            lp_solver:
                The LPSolver object.
            splits:
                List of the splits where each split is a dict:
                {"node": int, "neuron": int, "split_x": float, "upper": bool}

        Returns:
              The lp-solver constraint key.
        """

        if CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS:
            num_eq_pr_constraint = 2
            var_num = sip.nodes[0].in_size + sip.num_non_linear_neurons
        else:
            num_eq_pr_constraint = 1
            var_num = sip.nodes[0].in_size

        all_coeffs = np.zeros((len(splits)*num_eq_pr_constraint, var_num), dtype=np.float32)
        constants = np.zeros(len(splits)*num_eq_pr_constraint, dtype=np.float32)
        constr_types = [None] * len(splits)*num_eq_pr_constraint

        for i, split in enumerate(splits):

            node, neuron, split_x, upper = split["node"], split["neuron"], split["split_x"], split["upper"]

            constr_types[i] = 'G' if upper else 'L'
            coeffs = sip.get_neuron_bounding_equation(node, neuron, lower=not upper,
                                                      bias_sep_constraints=False)[0][0].cpu().detach().numpy()
            constants[i] = coeffs[-1] - split_x
            all_coeffs[i, :coeffs.shape[0] - 1] = coeffs[:-1]

            if CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS:

                constr_types[i + len(splits)] = 'G' if upper else 'L'
                coeffs_bias = sip.get_neuron_bounding_equation(node,
                                                               neuron,
                                                               lower=not upper,
                                                               bias_sep_constraints=True)[0][0].cpu().detach().numpy()
                constants[i + len(splits)] = coeffs_bias[-1] - split_x
                all_coeffs[i + len(splits), :coeffs_bias.shape[0] - 1] = coeffs_bias[:-1]

        constraints = (lp_solver.add_constraints(coeffs=all_coeffs, constants=constants, constr_types=constr_types))

        reshaped_constraints = []
        for i in range(len(splits)):
            reshaped_constraints.append([constraints[i + j*len(splits)] for j in range(num_eq_pr_constraint)])

        return reshaped_constraints

    def add_all_constrains(self, sip: SIP, solver: LPSolver, split_list: list):

        """
        Adds all constrains in the split list to the LP-solver.

        This method assumes that the LP-solver does not have any invalid constraints
        from previous branches.

        Args:
            sip:
                The SIP object.
            solver:
                The LPSolver.
            split_list:
                The list with splits from the old branch.
        """

        assert self.lp_solver_constraints is None, "Tried adding new constraints before removing old"

        if len(split_list) == 0:
            return

        self.lp_solver_constraints = Branch.add_constraints_to_solver(sip, solver, split_list)

    def update_constraints(self, sip: SIP, solver: LPSolver, old_split_list: list, old_constr_list: list):

        """
        Updates the constraints from the constraints of the last branch to the
        constraints of this branch.

        All constraints due to splits that are not in this branch are removed. We also
        re-add all constraints from splits in nodes after the earliest node with a
        removed or new split; this is done as the equation in SIP may have changed.
        All other constraints are kept as is.

        Args:
            sip:
                The SIP Object.
            solver:
                The LPSolver.
            old_split_list:
                The list with splits from the old branch.
            old_constr_list:
                The list with constraints from the old branch.
        """

        assert self.lp_solver_constraints is None, "Tried adding new constraints before removing old"

        old_constr_list = old_constr_list if old_constr_list is not None else []

        self.lp_solver_constraints = []
        min_node = self._get_earliest_changed_node_num(sip.num_nodes, old_split_list)
        re_add_idx = [i for i in range(len(old_split_list[:self.depth - 1])) if old_split_list[i]["node"] > min_node]
        self._update_lp_solver_constraints(sip, solver, old_constr_list, re_add_idx)

    def _get_earliest_changed_node_num(self, num_nodes: int, old_split_list: list) -> int:

        """
        Compares the split_list with old_split_list to find the earliest
        node where a split-constraint was removed.

        A depth-first-search is assumed, thus if self.depth > len(old_split_list),
        no split-constraints have been removed (descending into the three). On
        backtrack, all nodes from self.depth - 1 in len(old_split_list) are removed.

        Args:
            num_nodes:
                The number of nodes in the network.
            old_split_list:
                The list with splits from the old branch.
        Returns:
            The node-number of the earliest node with changed split-constraints.
        """

        min_node = num_nodes
        for i in range(self.depth - 1, len(old_split_list)):
            min_node = min(min_node, old_split_list[i]["node"])

        return min(min_node, self.split_list[-1]["node"])

    def _update_lp_solver_constraints(self,
                                      sip: SIP,
                                      solver: LPSolver,
                                      old_constr_list: list,
                                      re_add_idx: list):

        """
        Updates the LP-solver constraints.

        Any constraint from old_constr_list is removed and re-added to the solver
        as well as the last split in self.split_list.

        Args:
            sip:
                The SIP Object.
            solver:
                The LPSolver.
            old_constr_list:
                The list with constraints from the old branch.
            re_add_idx:
                A list of indices corresponding to constraints from old_constr_list
                that should be updated.
        """

        re_add_splits = [self.split_list[i] for i in re_add_idx]

        # Remove invalid constraints from backtrack and due to changes in SIP bounds
        invalid_constraints = []
        for i in range(self.depth - 1, len(old_constr_list)):
            invalid_constraints += old_constr_list[i]
        for i in re_add_idx:
            invalid_constraints += old_constr_list[i]

        solver.remove_constraints(invalid_constraints)

        updated_constraints = Branch.add_constraints_to_solver(sip, solver, re_add_splits + [self.split_list[-1]])

        self._lp_solver_constraints = old_constr_list[0:self._depth-1].copy() if old_constr_list is not None else []

        for i, new_constraint in enumerate(updated_constraints[:-1]):
            self._lp_solver_constraints[re_add_idx[i]] = new_constraint

        self.lp_solver_constraints.append(updated_constraints[-1])
