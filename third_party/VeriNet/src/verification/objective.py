
"""
A class representing the verification objective.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import List

import numpy as np
import torch

from verinet.neural_networks.verinet_nn import VeriNetNN
from verinet.sip_torch.sip import SIP

from verinet.verification.verifier_util import Status
from verinet.constraints.var import Var
from verinet.constraints.clp_constraint import CLPConstraint
from verinet.util.config import CONFIG
from verinet.verification.verifier_util import Branch
from verinet.verification.lp_solver import LPSolver


class Objective:

    # noinspection PyCallingNonCallable
    def __init__(self, input_bounds: np.array, output_size: int, model: VeriNetNN):

        """
        Args:
            input_bounds:
                An array with dimension Nx2 where N is the networks input
                dimensionality. The first column should contain the lower bounds and
                the second the upper.
            output_size:
                The number of output neurons.
            model:
                The VeriNetNN _model
        """

        self.model = model
        self.model.set_device(False)

        self._constraints = []
        self._torch_constraints_eq = []
        self._output_vars = None
        self._safe_constraints = None

        self._tensor_type = torch.DoubleTensor if model.uses_64bit else torch.FloatTensor

        self._active_lp_solver_constraints = None
        self._current_constraint_idx = None

        self._input_bounds = input_bounds.astype(np.float32)
        self._input_bounds_flat = self._input_bounds.reshape(-1, 2)

        if np.sum(self._input_bounds_flat[:, 0] > self._input_bounds_flat[:, 1]) > 0:
            raise ValueError("Got lower input bounds that are smaller than the upper bounds.")

        self._input_bounds_flat_torch = self._tensor_type(self._input_bounds_flat).to(device=model.device)

        # Calculate the input shape (including channel dimension, excluding batch dimension for 2D)
        self._input_shape = input_bounds.shape[:-1]
        if len(self._input_shape) == 2:
            self._input_shape = (1, *self._input_shape)  # Add channel
        if len(self._input_shape) == 4:
            self._input_shape = self._input_shape[1:]  # Remove batch dimension

        # Initialise the output variables used for constraints
        self._output_size = output_size
        self._output_vars = Var.factory(output_size)
        self._first_var_id, self._last_var_id = self._output_vars[0].id, self._output_vars[-1].id

    @property
    def input_bounds(self):
        return self._input_bounds

    @property
    def input_bounds_flat(self):
        return self._input_bounds_flat

    @property
    def input_bounds_flat_torch(self):
        return self._input_bounds_flat_torch

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_size(self):
        return np.prod(self.input_shape)

    @property
    def output_size(self):
        return self._output_size

    @property
    def safe_constraints(self):
        return self._safe_constraints

    @safe_constraints.setter
    def safe_constraints(self, val: list):
        self._safe_constraints = val

    @property
    def unsafe_constraints(self):
        return [i for i in range(len(self._constraints)) if i not in self.safe_constraints]

    @property
    def output_vars(self):
        return self._output_vars

    @property
    def num_constraints(self):
        return len(self._torch_constraints_eq)

    @property
    def current_constraint_idx(self):
        return self._current_constraint_idx

    @current_constraint_idx.setter
    def current_constraint_idx(self, val: int):
        self._current_constraint_idx = val

    @staticmethod
    def init_from_vnnlib(model: VeriNetNN,
                         vnnlib_path: str,
                         input_var_name: str = "X",
                         output_var_name: str = "Y",
                         input_shape: tuple = None):

        """
        Initialises the objective from a vnnlib file.

        Please see the doc in VNNLIBParser for more information about supported
        file formats.

        Args:
            model:
                The VeriNetNN model
            vnnlib_path:
                The path of the vnnlib file
            input_var_name:
                The name of the input variables in the vnnlib file
            output_var_name:
                The name of the output variables in the vnnlib file.
            input_shape:
                The shape of the model's input.

        Returns:
            The objective.
        """

        raise NotImplementedError

    # noinspection PyArgumentList,PyCallingNonCallable
    def add_constraints(self, constraints: List[CLPConstraint]):

        """
        Adds verification constraints.

        A network is considered safe for the given input if all valid
        inputs fulfill all given constraints.

        Args:
            constraints:
                A list with the constraints.
        """

        if isinstance(constraints, CLPConstraint):
            constraints = [constraints]

        self._constraints += constraints

        first, last = self._first_var_id, self._last_var_id
        self._torch_constraints_eq += [self._tensor_type(np.array(constr.as_arrays(first, last)))
                                       for constr in constraints]

    def remove_constraints(self, constraints: List[CLPConstraint]) -> List[bool]:

        """
        Removes verification constraints.

        Args:
            constraints:
                A list with the constraints
        Returns:
            A list with True/False values for each constraint indicating the given
            constraint was removed successfully.
        """

        success = []
        if isinstance(constraints, CLPConstraint):
            constraints = [constraints]

        for constraint in constraints:
            if constraint in self._constraints:
                idx = self._constraints.index(constraint)
                del self._constraints[idx]
                del self._torch_constraints_eq[idx]
                success += [True]
            else:
                success += [False]

        return success

    def grad_descent_loss(self, y: torch.Tensor) -> torch.Tensor:

        """
        Returns the loss for gradient descent.

        Args:
            y:
                The neural network output
        Returns:
            The loss function
        """

        constr_eqs = self._torch_constraints_eq[self._current_constraint_idx]

        loss = torch.zeros(1).to(device=self.model.device)
        for constr_eq in constr_eqs:
            constr_eq = constr_eq.to(device=self.model.device)
            # Clamping loss for satisfied condition to -0.1 instead of 0 to provide some "wiggle room"
            loss += torch.clamp((-y[0, :] * constr_eq[:-1]).sum() - constr_eq[-1], 0, 1e8)
        return loss

    def is_counter_example(self, y: np.array) -> bool:

        """
        Returns True if the output constraints are violated.

        Args:
            y:
                The output of the neural network.
        """

        for constr_eq in self._torch_constraints_eq[self._current_constraint_idx]:
            constr_eq = constr_eq.to(device=self.model.device)
            if (y[0, :] * constr_eq[:-1]).sum() + constr_eq[-1] <= 0:
                return False
        return True

    # noinspection PyUnresolvedReferences
    def get_summed_constraints(self) -> np.array:

        """
        Returns the sum of all non-safe constraint equations.

        The equation can be used in a heuristic to find a good split-candidate.

        Returns:
            An array with the sum of all constraint equations.
        """

        constr_coeffs = np.zeros(self._output_size)
        for i, constraint in enumerate(self._constraints):
            if not self._constraint_is_safe(i):
                terms = self._constraints[i].as_arrays(self._first_var_id, self._last_var_id)
                for term in terms:
                    constr_coeffs += term[:-1]

        return constr_coeffs

    def _constraint_is_safe(self, constr_idx: int):

        """
        Checks if the given constraint is safe.

        Args:
            constr_idx:
                The index of the constraint that should be checked.
        """

        return True if constr_idx in self._safe_constraints else False

    def find_potential_cex(self, branch: Branch, solver: LPSolver, sip: SIP) -> tuple:

        """
        Locates potential CEX under the SIP constraints

        Args:
            branch:
                The current branch.
            solver:
                The LPSolver object with branch constraints.
            sip:
                The SIP object with calculated bounds.
        Returns:
            A tuple bool, potential_cex, value where the bool = True indicates that
            all constraints have been checked while potential_cex is a vector that
            satisfies all constraints if found, otherwise None. Value is the
            value the constraint evaluated at potential_cex. Note that if
            the constraint has more than one equation, value is evaluated at the
            last equation.
        """

        while ((self._current_constraint_idx < len(self._constraints)) and
               (self._constraint_is_safe(self._current_constraint_idx))):
            self._current_constraint_idx += 1

        if self._constraints is None or len(self._constraints) <= self._current_constraint_idx:
            self._current_constraint_idx = 0
            return True, None, 0  # Finished checking all constraints

        else:
            constr_eqs = self._constraints[self._current_constraint_idx].as_arrays(self._first_var_id,
                                                                                   self._last_var_id)

            # Try simple lp-solver
            if CONFIG.USE_SIMPLE_LP and len(constr_eqs) == 1 and \
                    len(branch.split_list) == 0 and not CONFIG.USE_OPTIMISED_RELAXATION_CONSTRAINTS:

                potential_cex, value = self._find_potential_cex_simple(sip, False)

                return False, potential_cex, value

            # Start advanced search
            use_optimised_relaxations = (CONFIG.USE_OPTIMISED_RELAXATION_CONSTRAINTS and
                                         sip.has_cex_optimisable_relaxations)
            potential_cex, value = self._find_potential_cex(solver, sip, use_optimised_relaxations)

            return False, potential_cex, value

    # noinspection PyCallingNonCallable
    def _find_potential_cex(self, solver: LPSolver, sip: SIP,
                            use_optimised_relaxation_constraints: bool) -> tuple:

        """
        Locates potential CEX under the SIP constraints

        Args:
            solver:
                The LPSolver object
            sip:
                The SIP object
            use_optimised_relaxation_constraints:
                If true, a second run is performed with relaxations optimised
                for the model-values computed by spurious counter examples.
        Returns:
            The potential cex and corresponding value of the upper bound. 
        """

        self._active_lp_solver_constraints, value, result = \
            self._get_output_constraints(solver, sip, CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS)

        if not result:
            return None, 0

        else:

            if use_optimised_relaxation_constraints:

                for _ in range(CONFIG.NUM_ITER_OPTIMISED_RELAXATIONS):
                    potential_cex = self._tensor_type(solver.get_assigned_input_values()).to(device=self.model.device)
                    out = self.model.forward(potential_cex.view((1, *self.input_shape)), cleanup=False)

                    if self.is_counter_example(out[0]):
                        return potential_cex, value

                    self.calc_optimised_relaxations(sip)

                    sip.set_optimised_relaxations()
                    new_constraints, value, result = self._get_output_constraints(solver, sip, False)
                    self._active_lp_solver_constraints += new_constraints
                    sip.set_non_parallel_relaxations()

                    for node in self.model.nodes:
                        node.value = None
                        node.input_value = None

                    if not result:
                        return None, 0

            potential_cex = self._tensor_type(solver.get_assigned_input_values()).to(device=self.model.device)
            return potential_cex, value

    def calc_optimised_relaxations(self, sip: SIP):

        """
        Adjusts the relaxations of the sip nodes to relaxations optimised for
        the values calculated by the neural network.

        Uses the last calculated values stored in model.

        Args:
            sip:
                The SIP object
        """

        for i in range(len(sip.nodes)):

            sip_node = sip.nodes[i]
            nn_node = self.model.nodes[i]

            if not sip_node.is_linear:
                sip_node.calc_optimised_relaxations(nn_node.input_value[0].reshape(-1))

    # noinspection PyCallingNonCallable,PyMethodMayBeStatic
    def _get_output_constraints(self, solver: LPSolver, sip: SIP, use_bas_separated_constraints: bool = False) -> tuple:

        """
        Returns the lp-output constraints for self._current_constraint_idx

        Args:
            solver:
                The LPSolver object
            sip:
                The SIP object
        Returns:
            The lp-solver constraints, value of the last constraint evaluated at
            the potential cex and the result of the lp-call.
        """

        constr_eqs = self._constraints[self._current_constraint_idx].as_arrays(self._first_var_id,
                                                                               self._last_var_id)

        assert len(constr_eqs) > 0, "Expected at least one constr_eqs"
        
        lp_solver_constraint = []
        coeffs = None
        constant = 0

        for constr_eq in constr_eqs:

            # Add constraints from SIP
            constr_eq = constr_eq.astype(np.float32)

            coeffs = sip.convert_output_bounding_equation(torch.Tensor(constr_eq[:-1]).view(1, -1),
                                                          lower=False).cpu().numpy()[0]
            constant = coeffs[-1] + constr_eq[-1]

            lp_solver_constraint.append(solver.add_constraints(coeffs=coeffs[:-1],
                                                               constants=np.array([constant]),
                                                               constr_types=['G'])[0])

            if use_bas_separated_constraints:
                coeffs_bias_sep = sip.convert_output_bounding_equation(torch.Tensor(constr_eq[:-1]).view(1, -1),
                                                                       lower=False,
                                                                       bias_sep_constraints=True).cpu().numpy()[0]
                constant_bias_sep = coeffs_bias_sep[-1] + constr_eq[-1]

                lp_solver_constraint.append(solver.add_constraints(coeffs=coeffs_bias_sep[:-1],
                                                                   constants=np.array([constant_bias_sep]),
                                                                   constr_types=['G'])[0])

            if CONFIG.PERFORM_LP_MAXIMISATION:
                solver.maximise_objective(coeffs[:sip.nodes[0].in_size], constant)

        result = solver.solve()

        if result:
            potential_cex = self._tensor_type(solver.get_assigned_input_values())
            value = torch.sum(self._tensor_type(coeffs[:-1]) * potential_cex) + constant
        else:
            value = 0

        return lp_solver_constraint, value, result

    # noinspection PyCallingNonCallable
    def _find_potential_cex_simple(self, sip: SIP,
                                   use_optimised_relaxation_constraints: bool) -> tuple:

        """
        Locates potential CEX under the SIP constraints

        A simplified version that does not use the LP-Solver. Note that constraints
        added to the LP-Solver are not considered, thus this method is typically
        only used when no hidden-node splits are performed yet.

        Args:
            sip:
                The SIP object
            use_optimised_relaxation_constraints:
                If true, a second run is performed with relaxations optimised
                for the model-values computed by spurious counter examples.
        Returns:
            (potential_cex, max_val), if no potential_cex was found None is returned
            instead.
        """

        constr_eqs = self._constraints[self._current_constraint_idx].as_arrays(self._first_var_id,
                                                                               self._last_var_id)
        if len(constr_eqs) > 1:
            raise ValueError("Simple method can only handle one constraint equation")

        constr_eq = constr_eqs[0]

        # Add constraints from SIP
        constr_eq = constr_eq.astype(np.float32)

        coeffs = sip.convert_output_bounding_equation(self._tensor_type(constr_eq[:-1]).view(1, -1), lower=False)[0]
        constant = coeffs[-1] + constr_eq[-1]

        potential_cex, max_val = self.maximise_eq(coeffs[:-1], sip.get_bounds_concrete_post(0))
        max_val += constant

        if max_val < 0:
            return None, 0
        else:
            if use_optimised_relaxation_constraints:

                out = self.model.forward(potential_cex.view((1, *self.input_shape)), cleanup=False)

                if self.is_counter_example(out[0]):
                    return potential_cex

                self.calc_optimised_relaxations(sip)

                sip.set_optimised_relaxations()
                coeffs = sip.convert_output_bounding_equation(self._tensor_type(constr_eq[:-1]).view(1, -1),
                                                              lower=False)[0]
                constant = coeffs[-1] + constr_eq[-1]

                potential_cex, max_val = self.maximise_eq(coeffs[:-1], sip.get_bounds_concrete_post(0))
                max_val += constant
                sip.set_non_parallel_relaxations()

                if max_val < 0:
                    return None, 0

            return potential_cex, max_val

    # noinspection PyArgumentList
    @staticmethod
    def maximise_eq(eq: torch.Tensor, bounds: torch.Tensor) -> tuple:

        """
        Maximises the given equation where each variable is bounded by bounds.

        Args:
            eq:
                A tensor of length N
            bounds:
                A Nx2 tensor with the lower and upper bounds of the eq variables.
        Returns:
            The eval point and max val.
        """

        eval_point = torch.zeros((bounds.shape[0], ), dtype=bounds.dtype).to(eq.device)

        eval_point[eq < 0] = bounds[eq < 0, 0]
        eval_point[eq > 0] = bounds[eq > 0, 1]

        max_val = float(torch.sum(eq * eval_point))

        return eval_point, max_val

    def finished_constraint(self, solver: LPSolver, status: Status):

        """
        Stores safe constraint indices in self._safe_constraints

        Args:
            solver:
                The LPSolver
            status:
                The Status after the last run of VeriNet
        """

        if self._active_lp_solver_constraints is not None:
            solver.remove_constraints(self._active_lp_solver_constraints)
            self._active_lp_solver_constraints = None

        if status == Status.Safe:
            self._safe_constraints.append(self._current_constraint_idx)

        self._current_constraint_idx += 1

    def cleanup(self, solver: LPSolver):

        """
        Used to reset settings used in DeepSplit

        Args:
            solver  : The LP-solver
        """

        self._current_constraint_idx = 0
        self._safe_constraints = []
        if self._active_lp_solver_constraints is not None:
            solver.remove_constraints(self._active_lp_solver_constraints)
            self._active_lp_solver_constraints = None
