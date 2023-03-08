
"""
This file contains the LPSolver part of VeriNet

The LPSolver uses the symbolic bounds and the Xpress solver to verify properties as
Safe, or to produce candidates for counter examples.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import sys
import os

import numpy as np

from verinet.util.config import CONFIG

# Disable print informing about community license
sys.stdout = open(os.devnull, 'w')
import xpress as xp
sys.stdout.close()
sys.stdout = sys.__stdout__


# noinspection PyUnresolvedReferences
class LPSolver:

    """
    The LPSolver class combines the symbolic bounds from SIP and the Xpress LP-Solver
    to verify properties as safe or produce candidates for counter examples.
    """

    def __init__(self, input_size: int):

        """
        Args:
            input_size:
                The number of input neurons.
        """

        self._solver = xp.problem()
        self._solver.setControl('outputlog', 0)
        self._solver.setControl('presolve', CONFIG.USE_LP_PRESOLVE)  # Empirically determined to be faster
        self._solver.setControl('threads', 1)
        self._solver.setControl('deterministic', 1)

        self._input_size = input_size
        self._input_variables = None
        self._bias_variables = None
        self._init_variables()

    @property
    def num_bias_vars(self) -> int:

        if self._bias_variables is None:
            return 0
        else:
            return self._bias_variables.shape[0]

    @property
    def variables(self):

        if self._input_variables is None:
            raise ValueError("Input variables not initialised.")
        elif self._bias_variables is None:
            return self._input_variables
        else:
            return np.concatenate((self._input_variables, self._bias_variables))

    def __del__(self):

        """
        Silently deletes the xpress solver object.

        A bug occurred where deleting the xpress solver object printed a blank line to
        terminal. This method is a crude fix for the bug.
        """

        sys.stdout = open(os.devnull, 'w')
        del self._solver
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def _init_variables(self):

        """
        Initializes the input and error variables.
        """

        self._remove_variables()
        self._input_variables = np.array([xp.var(lb=0, ub=1) for _ in range(self._input_size)])
        self._solver.addVariable(self._input_variables)

    def add_bias_variables(self, num: int):

        """
        Adds bias variables to the verification problem.
        """

        new_variables = np.array([xp.var(lb=0, ub=1) for _ in range(num)])
        self._solver.addVariable(new_variables)

        if self._bias_variables is None:
            self._bias_variables = new_variables
        else:
            self._bias_variables = np.concatenate((self._bias_variables, new_variables))

    # noinspection PyArgumentList
    def _remove_variables(self):

        """
        Removes all variables.
        """

        if self._input_variables is not None:
            self._solver.delVariable(self._input_variables)
            self._input_variables = None
        if self._bias_variables is not None:
            self._solver.delVariable(self._bias_variables)
            self._bias_variables = None

    def solve(self) -> bool:

        """
        Solves the system with current constraints.

        Returns:
            True if the system is feasible, else False.
        """

        self._solver.solve()

        if self._solver.getProbStatus() == 1:
            return True  # Found an assignment
        elif self._solver.getProbStatus() == 2:
            return False  # Infeasible
        else:
            raise LPSolverException("Unexpected solver status")

    # noinspection PyArgumentList
    def set_input_bounds(self, input_bounds: np.array):

        """
        Sets the input variable bounds.

        Args:
            input_bounds:
                A Nx2 array where N is the networks input dimension, the first column
                contains the lower bounds and the second contains the upper bounds.
        """

        if self._input_variables is None:
            raise VariablesNotInitializedException("set_input_bounds() called before input variables where initialized")

        self._solver.chgbounds(self._input_variables[:self._input_size], ['L'] * self._input_size, input_bounds[:, 0])
        self._solver.chgbounds(self._input_variables[:self._input_size], ['U'] * self._input_size, input_bounds[:, 1])

    # noinspection PyArgumentList
    def get_assigned_input_values(self) -> np.array:

        """
        Returns the currently assigned input values.

        Returns:
            input_values or None if no input values are assigned.
        """

        return np.array(self._solver.getSolution(self._input_variables))

    def add_constraints(self, coeffs: np.array, constants: np.array, constr_types: list) -> xp.constraint:

        """
        Adds an LP-constraint to the solver.

        Adds constraints with respect to self.variables. The variables are
        multiplied by the provided coeffs and the constant is added.

        Tested solutions to improve performance (this is a bottleneck):

        1) Tried using self._solver.addConstraint(), but self._solver.addrows() turned
        out to be 3x faster.

        Args:
            coeffs:
                A MxN array with the coefficients of the input variables where M
                is the number of constraints and N is the number of input and error
                variables
            constants:
                An array of length M with the constant terms (lhs constants) where
                M is the number of constraints
            constr_types:
                 A list of length M with the constraint type Constraint('L') for LQ
                 of 'G' for GQ. M is the number of constraints.

        Returns:
            The constraint key
        """

        coeffs = np.atleast_2d(coeffs)
        var_num = coeffs.shape[1]
        num_constraints = len(constants)
        mstart = [i*var_num for i in range(num_constraints + 1)]
        mclind = list(self.variables[:var_num]) * num_constraints

        self._solver.addrows(qrtype=constr_types,
                             rhs=-constants,
                             mstart=mstart,
                             mclind=mclind,
                             dmatval=coeffs.reshape(-1))

        return self._solver.getConstraint()[-num_constraints:]

    def get_all_constraints(self):

        """
        Fetches all constraints.

        Returns:
            A list of all constraints.
        """

        return self._solver.getConstraint()

    def remove_constraints(self, constraints: xp.constraint):

        """
        Removes the given constraint from solver.

        Args:
            constraints: The key of the constraint.
        """

        if len(constraints) > 0:
            self._solver.delConstraint(constraints)

    def remove_all_constraints(self):

        """
        Removes all constraint from solver.
        """

        if len(self._solver.getConstraint()) > 0:
            self._solver.delConstraint(self._solver.getConstraint())

    def maximise_objective(self, coeffs: np.array, constant: np.array):

        """
        Sets a maximisation objective.

        The objective function is sum(var.dot(coeffs)) + constant.

        Args:
            coeffs:
                A N array with the coefficients of the input variables where N is the
                number of input and error variables.
            constant:
                The constant part of the expression.
        """

        expr = xp.Sum(self._input_variables[i] * coeffs[i] for i in range(coeffs.shape[0])) + constant
        self._solver.setObjective(expr, sense=xp.maximize)

    def minimise_objective(self, coeffs: np.array, constant: np.array):

        """
        Sets a maximisation objective.

        The objective function is sum(var.dot(coeffs)) + constant.

        Args:
            coeffs:
                A N array with the coefficients of the input variables where N is the
                number of input and error variables.
            constant:
                The constant part of the expression.
        """

        expr = xp.Sum(self._input_variables[i] * coeffs[i] for i in range(coeffs.shape[0])) + constant
        self._solver.setObjective(expr, sense=xp.minimize)


class LPSolverException(Exception):
    pass


class VariablesNotInitializedException(LPSolverException):
    pass
