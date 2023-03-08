
"""
This file contains the LPSolver part of the algorithm

The LPSolver uses the symbolic bounds and the Gurobi solver to verify properties as Safe, or to produce candidates for
counter examples.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import numpy as np
import gurobipy as grb

from src.algorithm.esip import ESIP


class LPSolver:

    """
    The LPSolver class combines the symbolic bounds from ESIP and Gurobi LPSolver to verify properties as safe
    or produce candidates for counter examples
    """

    def __init__(self, input_size: int, output_size: int):

        """
        Args:
              input_size    : The number of input nodes
              output_size   : The number of output nodes
        """
        self._disable_log()

        self._grb_solver = grb.Model("NN")
        self._input_size = input_size
        self._output_size = output_size

        self._input_variables = None
        self._output_variables = None

        self._init_variables()

    @property
    def grb_solver(self):
        return self._grb_solver

    @property
    def input_variables(self):
        return self._input_variables

    @property
    def output_variables(self):
        return self._output_variables

    def _init_variables(self):

        """
        Initializes the Gurobi input and output variables

        The default bounds on the variables are from the ESIP object. If output bounds are also given the
        these are used to refine the bounds.
        """

        self._remove_variables()

        self._input_variables = (self._grb_solver.addVars(range(self._input_size), lb=-grb.GRB.INFINITY,
                                                          ub=grb.GRB.INFINITY,
                                                          vtype=grb.GRB.CONTINUOUS, name="Input"))
        self._output_variables = (self._grb_solver.addVars(range(self._output_size), lb=-grb.GRB.INFINITY,
                                                           ub=grb.GRB.INFINITY,
                                                           vtype=grb.GRB.CONTINUOUS, name="Output"))

        self._grb_solver.update()

    # noinspection PyArgumentList
    def _remove_variables(self):

        """
        Removes all variables initialized from nn_bounds
        """

        if self._input_variables is not None:
            self.grb_solver.remove(self._input_variables)
            self._input_variables = None
        if self._output_variables is not None:
            self.grb_solver.remove(self._output_variables)
            self._node_variables = None

        self._grb_solver.update()

    def solve(self) -> bool:

        """
        Solves the system with current constraints

        All variables are initialized to the given _input_bounds and _output_bounds. Symbolic interval propagation
        is used to further refine the output bounds and to add the linear constraints on he output resulting from
        the symbolic intervals.

        Returns:
            True if the system is feasible, else False
        """

        # Uncommenting this line avoids gurobi _status code 4(INF_OR_UNBD)
        # self._grb_solver.setParam("DualReductions", 0)

        # Using dual simplex as it is numerically stable and the fastest for our tests
        self._grb_solver.setParam("Method", 0)

        self._grb_solver.optimize()

        if self._grb_solver.status == 2:  # Found an assignment
            return True
        elif self._grb_solver.status == 3:  # Infeasible system
            return False
        else:
            raise UnexpectedGurobiStatusException(f"Gurobi _status: {self._grb_solver._status}")

    # noinspection PyArgumentList
    def set_variable_bounds(self, bounds: ESIP, output_bounds: np.array=None, set_input: bool=True):

        """
        Sets the variable bounds using bounds from ESIP, and possibly output_bounds

        Args:
            bounds          : The ESIP object
            output_bounds   : A Nx2 array-like structure with the lower output bounds in the first and column
                              and upper in the second. The tightest bounds from ESIP and this array will
                              be used
            set_input       : If False, the input variables aren't adjusted
        """

        if self._input_variables is None or self._output_variables is None:
            raise VariablesNotInitializedException("set_input_bounds() called before input variables where initialized")

        if set_input:
            input_bounds_lower = bounds.bounds_concrete[0][:, 0]
            input_bounds_upper = bounds.bounds_concrete[0][:, 1]

            for node_num, var in enumerate(self._input_variables.select()):
                var.lb, var.ub = input_bounds_lower[node_num], input_bounds_upper[node_num]

        output_bounds_lower = bounds.bounds_concrete[-1][:, 0].copy()
        output_bounds_upper = bounds.bounds_concrete[-1][:, 1].copy()

        if output_bounds is not None:
            # Refine the output bounds using the given output_bounds array
            better_lower_idx = output_bounds[:, 0] > output_bounds_lower
            better_upper_idx = output_bounds[:, 1] < output_bounds_upper
            output_bounds_lower[better_lower_idx] = output_bounds[better_lower_idx, 0]
            output_bounds_upper[better_upper_idx] = output_bounds[better_upper_idx, 1]

        for node_num, var in enumerate(self._output_variables.select()):
            var.lb, var.ub = output_bounds_lower[node_num], output_bounds_upper[node_num]

        self._grb_solver.update()

    # noinspection PyArgumentList
    def get_assigned_values(self) -> tuple:

        """
        Returns the currently assigned input and output values

        Returns None if no values are assigned.

        Returns:
            (input_values, output_values)
        """

        try:
            input_values = [var.x for var in self._input_variables.select()]
        except AttributeError:
            # Values not assigned, solve() probably hasn't been called
            input_values = None

        try:
            output_values = [var.x for var in self._output_variables.select()]
        except AttributeError:
            # Values not assigned, solve() probably hasn't been called
            output_values = None

        return np.array(input_values), np.array(output_values)

    # noinspection PyMethodMayBeStatic
    def _disable_log(self):

        """
        Remove the automatically created Gurobi log file and disable future logging.
        """

        # Get rid of log file
        grb.setParam("OutputFlag", 0)
        grb.setParam("LogFile", "")
        try:
            os.remove("./gurobi.log")
        except FileNotFoundError:
            pass


class LPSolverException(Exception):
    pass


class VariablesNotInitializedException(LPSolverException):
    pass


class UnexpectedGurobiStatusException(LPSolverException):
    pass
