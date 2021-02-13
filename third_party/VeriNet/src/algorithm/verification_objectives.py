
"""
This file contains objectives for the verification algorithms in VeriNet.

Each _verification_objective contains callback functions for gradient descent loss, determining if a counter-example is
valid, and gurobi constraints.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import numpy as np
import gurobipy as grb
from typing import Callable

from src.algorithm.esip import ESIP
from src.algorithm.lp_solver import LPSolver
from src.algorithm.verinet_util import Status


class VerificationObjective:

    """
    A abstract class used by by the gradient descent function for finding counter examples in VeriNet to determine
    the loss and the termination criteria
    """

    def __init__(self, input_bounds: np.array, output_bounds: np.array = None, output_size: int=None):

        """
        Args:
            input_bounds    : Bounds on the input of the network. The first dimensions should be the same as the input
                              to the network, the last dimension should contain the lower bounds in the first axis,
                              and the upper bounds in the second. Will be cast to np.float32, since this is
                              standard for neural networks.
            output_bounds   : Bounds on the output of the network given as a Nx2 array, can be None if not relevant.
            output_size     : The number of output nodes, only needed when output bounds is None.
        """

        # Used to temporarily store lp_solver settings for the cleanup function.
        self.vars = None
        self.constraints = None

        input_bounds = input_bounds.astype(np.float32)
        self._input_bounds = input_bounds
        self._input_bounds_flat = input_bounds.reshape(-1, 2)
        self._output_bounds = output_bounds
        self._safe_classes = None

        self._input_shape = input_bounds.shape[:-1]
        if len(self._input_shape) == 2:
            self._input_shape = (1, *self._input_shape)  # Add channel
        if len(self._input_shape) == 4:
            self._input_shape = self._input_shape[1:]  # Remove batch dimension

        if output_bounds is not None:
            self._output_size = int(np.prod(output_bounds.shape[:-1]))
        elif output_size is not None:
            self._output_size = output_size
        else:
            raise VerificationObjectiveException("output_bounds or output_size should be given")

    @property
    def input_bounds(self):
        return self._input_bounds

    @property
    def input_bounds_flat(self):
        return self._input_bounds_flat

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
    def output_bounds(self):
        return self._output_bounds

    @property
    def safe_classes(self):
        return self._safe_classes

    def is_safe(self, bounds: ESIP) -> bool:

        """
        Can be implemented to check if the verification problem can be determined before running the LPSolver,
        using only the calculated bounds. This will speed up the algorithm, but isn't necessary.

        Args:
            bounds: The ESIP object
        """

        return False

    def output_refinement_weights(self, bounds: ESIP) -> np.array:

        """
        Should return an array with the importance weight of the different output for determining which output to
        split on

        Args:
            bounds: The ESIP object
        Returns:
            An Mx2 array (Number of outputs) with weights for each outputs lower bounds in the first column and
            upper bounds in the second column.
        """

        raise NotImplementedError("output_refinement_weights() not implemented in subclass")

    def grad_descent_losses(self, lp_output: np.array, bounds: ESIP) -> Callable:

        """
        Should return the loss function for gradient descent to find counter-examples

        Example:
            The returned loss function will usually be the correct class minus the target class

        Args:
            lp_output   : The values of the output variables from the lp solver
            bounds      : The ESIP object

        Returns:
            The loss function
        """

        raise NotImplementedError("grad_descent_losses() not implemented in subclass")

    # noinspection PyUnusedLocal
    def is_counter_example(self, y: np.array) -> bool:

        """
        Returns True if gradient descent the given example is a valid counter-example

        Args:
            y: The output of the neural network
        """

        raise NotImplementedError("is_counter_example() not implemented in subclass")

    def initial_settings(self, solver: LPSolver, bounds: ESIP, safe_classes: list):

        """
        Should do initial setup and adjust lp solver constraints.

        This function is called at the beginning of each branch and should add constraints/variables to the
        LPSolver using the current bounds object. It can be assumed that the bounds object won't change until
        next time this function is called for the next branch.

        Args:
            solver              : The LPSolver object
            bounds              : The ESIP object
            safe_classes        : A list of classes that have been determined as safe in previous branches
        """

        raise NotImplementedError("initial_settings() not implemented in subclass")

    # noinspection PyTypeChecker
    def configure_next_potential_counter(self, solver: LPSolver, bounds: ESIP) -> bool:

        """
        Should configure the LPSolver for the next potential counter in the same split.

        This function is used to search for counter examples using several different LPSolver settings in the same
        split. Can for example be used to find adversarial classes by looking at one target class at a time.

        This function is called repeatedly as long as it returns true. After each call VeriNet does a run of the
        LPSolver to determine if the system is Safe/ Unsafe.

        Args:
            solver              : The LPSolver object
            bounds              : The ESIP object
        """

        raise NotImplementedError("configure_next_potential_counter() not implemented in subclass")

    def finished_potential_counter(self, solver: LPSolver, status: Status):

        """
        A callback function called from VeriNet after a run with the settings from the last time
        configure_next_potential_counter

        Can for example be used to keep track of safe classes returned stored in self._safe_classes used to guide
        the LPSolver

        Args:
            solver  : The LPSolver
            status  : The _status after the last LP run of VeriNet
        """

        pass

    # noinspection PyArgumentList
    def cleanup(self, solver: grb.Model):

        """
        Used to remove all settings set by initial_settings()

        Args:
            solver  : The gurobi solver
        """

        if self.vars is not None:
            solver.remove(self.vars)
            self.vars = None
        if self.constraints is not None:
            for constr in self.constraints:
                solver.remove(constr)
                solver.update()

            self.constraints = None

        self._safe_classes = []


class LocalRobustnessObjective(VerificationObjective):

    """
    Used to calculate the loss and termination criteria for local robustness properties.
    """

    def __init__(self, correct_class: int, input_bounds: np.array, output_bounds: np.array=None,
                 output_size: int=None):

        """
        Args:
            correct_class   : The index of the correct class
            input_bounds    : Bounds on the input of the network. The first dimensions should be the same as the input
                              to the network, the last dimension should contain the lower bounds in the first axis,
                              and the upper bounds in the second.
            output_bounds   : Bounds on the output of the network given as a Nx2 array, can be None if not relevant
            output_size     : The number of output nodes, only needed when output bounds is None.
        """

        super().__init__(input_bounds, output_bounds, output_size)

        self.correct_class = correct_class
        self._safe_classes = []

        self.potential_counters = []
        self.current_potential_counter = None

    def is_safe(self, bounds: ESIP) -> bool:

        """
        Returns true if the correct class minimum value is larger than all other classes classes maximum

        Args:
            bounds: The ESIP object
        """

        return self.potential_counter(bounds).sum() == 0

    # noinspection PyUnresolvedReferences
    def potential_counter(self, bounds: ESIP) -> np.array:

        """
        Finds the classes that are potential counter examples.

        Args:
            bounds: The ESIP object
        Returns:
            A boolean array, where index i is true if class
        """

        potential_counter = (bounds.bounds_concrete[-1][:, 1] >=
                             bounds.bounds_concrete[-1][self.correct_class, 0])

        potential_counter[self.correct_class] = 0
        for safe_class in self.safe_classes:
            potential_counter[safe_class] = 0

        return potential_counter

    # noinspection PyUnresolvedReferences
    def output_refinement_weights(self, bounds: ESIP) -> np.array:

        """
        Returns an array with the importance weights for refinement

        The potential adversarial classes have weight 1 for the upper bounds, while to correct class has weight equal
        to the number of potential adversarial classes for the lower bound. This is because tightening the bounds of
        the correct class will affect all potential adversarial classes. The other weights are 0.

        Args:
            bounds: The ESIP object
        Returns:
            An Mx2 array (Number of outputs) with weights for each outputs lower bounds in the first column and
            upper bounds in the second column.
        """

        potential_counter = self.potential_counter(bounds).nonzero()[0]
        output_weights = np.zeros((bounds.layer_sizes[-1], 2), dtype=float)
        num_potential_counters = len(potential_counter)

        output_weights[potential_counter, 1] = 1
        output_weights[self.correct_class, 0] = num_potential_counters

        return output_weights

    def grad_descent_losses(self, lp_output: torch.Tensor, bounds: ESIP) -> Callable:

        """
        Returns the loss function for gradient descent.

        Args:
            lp_output   : The values of the output variables from the lp solver
            bounds      : The ESIP object

        Returns:
            A generator with the loss function
        """

        return lambda y: y[0, self.correct_class] - y[0, self.current_potential_counter]

    def is_counter_example(self, y: np.array) -> bool:

        """
        Returns True if the output for another class is larger than the output for the correct class.

        Args:
            y: The output of the neural network
        """

        return (y[0, self.correct_class] <= y[0, :]).sum() > 1

    # noinspection PyArgumentList,PyUnresolvedReferences
    def initial_settings(self, solver: LPSolver, bounds: ESIP, safe_classes: list):

        """
        Does initial setup and adjusts lp solver constraints

        For this verification problem, we use ESIP to identify potential counter example and constrain the output
        upper bound in the LPSolver for the correct class

        Args:
            solver              : The LPSolver object
            bounds              : The ESIP object
            safe_classes        : A list of classes that have been determined as safe in previous branches
        """

        self._safe_classes = safe_classes

        potential_counter = np.argwhere(self.potential_counter(bounds))

        if len(potential_counter) == 0:
            return
        else:
            potential_counter = potential_counter.reshape(-1)

        potential_counter_sorted_idx = bounds.bounds_concrete[-1][potential_counter, 1].argsort()
        self.potential_counters = list(potential_counter[potential_counter_sorted_idx])

        # Correct class maximum can't be larger than the maximum of target class
        potential_counter_max = bounds.bounds_concrete[-1][potential_counter, 1].max()
        if potential_counter_max < solver.output_variables[self.correct_class].ub:
            solver.output_variables[self.correct_class].ub = potential_counter_max

    def configure_next_potential_counter(self, solver: LPSolver, bounds: ESIP) -> bool:

        """
        Configures the LPSolver for the next potential counter

        Chooses the next potential counter and adds the necessary the relevant constraints to the LPSolver

        Args:
            solver              : The LPSolver object
            bounds              : The ESIP object
        Returns:
            True if potential counter we have a potential counter, else False
        """

        assert self.constraints is None, "Tried adding new constraints before removing old"

        if len(self.potential_counters) > 0:

            self.constraints = []

            self.current_potential_counter = self.potential_counters.pop()

            input_variables = solver.input_variables.select()
            bounds_symbolic = bounds.bounds_symbolic[-1]

            eq = (bounds_symbolic[self.current_potential_counter, :] -
                  bounds_symbolic[self.correct_class, :])

            error = (bounds.error_matrix[-1][self.current_potential_counter, :] -
                     bounds.error_matrix[-1][self.correct_class, :])

            eq[-1] += np.sum(error[error > 0])
            constr = (grb.LinExpr(eq[:-1], input_variables) + eq[-1] >= 0)

            self.constraints.append(solver.grb_solver.addConstr(constr))

            solver.grb_solver.update()
            return True

        else:
            return False

    def finished_potential_counter(self, solver: LPSolver, status: Status):

        """
        A callback function called from VeriNet after a run with the settings from the last time
        configure_next_potential_counter

        Keeps track of classes verified as safe

        Args:
            solver  : The LPSolver
            status  : The _status after the last LP run of VeriNet
        """

        if self.constraints is not None:
            for constr in self.constraints:
                solver.grb_solver.remove(constr)
            self.constraints = None
        solver.grb_solver.update()

        if status == Status.Safe:
            self.safe_classes.append(self.current_potential_counter)
        self.current_potential_counter = None

    def cleanup(self, solver: grb.Model):

        """
        Used to remove all settings set by initial_settings()

        Args:
            solver  : The gurobi solver
        """

        super().cleanup(solver)
        self.potential_counters = []
        self.current_potential_counter = None


class VerificationObjectiveException(Exception):
    pass
