"""
The main code for verification of neural networks

Author: Patrick Henriksen <patrick@henriksen.as>
"""

# noinspection PyUnresolvedReferences
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Callable, Optional
from collections import deque

from src.algorithm.lp_solver import LPSolver
from src.algorithm.esip import ESIP, BoundsException
from src.algorithm.verification_objectives import VerificationObjective
from src.algorithm.verinet_util import Status, Branch


class VeriNetWorker:

    def __init__(self,
                 model: nn,
                 verification_objective: VerificationObjective,
                 no_split: bool = False,
                 gradient_descent_intervals: int = 5,
                 gradient_descent_max_iters: int = 5,
                 gradient_descent_step: float = 1e-1,
                 gradient_descent_min_loss_change: float = 1e-2,
                 verbose=True
                 ):

        """
        Args:
            model                           : The torch neural network, the requires_grad parameters of this model
                                              might be changed.
            verification_objective          : The VerificationObjective
            no_split                        : If true no splitting is done
            verbose                         : If true information is printed at each branch
            gradient_descent_intervals      : Gradient descent performed to find counter example each number of
                                              intervals. Should be >0 0.
            gradient_descent_max_iters      : The number of iterations used in gradient descent to find a true counter
                                              example from a _lp_solver counter example
            gradient_descent_step           : The step size of the gradient descent used to find a true counter example
                                              from a _lp_solver counter example
            gradient_descent_min_loss_change: The minimum amount of change in loss from last iteration to keep trying
                                              gradient descent
        """

        self._model = model
        self._verification_objective = verification_objective
        self._no_split = no_split
        self._gradient_descent_intervals = gradient_descent_intervals
        self._gradient_descent_max_iters = gradient_descent_max_iters
        self._gradient_descent_step = gradient_descent_step
        self._gradient_descent_min_loss_change = gradient_descent_min_loss_change
        self._verbose = verbose

        self._status = Status.Undecided
        self._counter_example: torch.Tensor = None

        self._lp_solver: LPSolver = None
        self._bounds: ESIP = None

        self._branches = deque([])

        self._set_parameters_requires_grad(model=model, requires_grad=False)

        self.max_depth = 0
        self.branches_explored = 0

    @property
    def counter_example(self) -> torch.Tensor:
        return self._counter_example

    @property
    def status(self) -> Status:
        return self._status

    @property
    def bounds(self) -> ESIP:
        return self._bounds

    def _init_bounds(self):

        """
        Initializes the ESIP object
        """

        try:
            self._bounds = ESIP(self._model, self._verification_objective.input_shape)
        except BoundsException as e:
            raise VeriNetException("Error initializing ESIP in VeriNet") from e

    def _configure_lp_solver(self):

        """
        Initializes the LPSolver, sets the input/output bounds and adds constraints from ESIP
        """

        if self._bounds is None:
            raise VeriNetException("The bounds should be calculated before initializing the LPSolver")

        if self._lp_solver is None:
            self._lp_solver = LPSolver(self._verification_objective.input_size,
                                       self._verification_objective.output_size)
            self._lp_solver.set_variable_bounds(self._bounds, set_input=True)
        else:
            self._lp_solver.set_variable_bounds(self._bounds, set_input=False)

    def verify(self, start_branch: Branch, finished_flag: Optional[multiprocessing.Event],
               needs_branches: Optional[Callable], put_queue: Optional[Callable], queue_depth: int) -> Optional[Status]:

        """
        Runs the main algorithm for verification of the given verification _verification_objective

        Args:
            start_branch     : The first branch
            finished_flag    : If set, the worker will abort. If None this parameter is disregarded.
            needs_branches   : A callable that should return true if the queue needs more branches. If None
                               branches will never be added to queue
            put_queue        : A function to put new branches into the working queue. If None
                               branches will never be added to queue
            queue_depth      : If the depth difference between a branch and the deepest branch is more than this,
                               the branch will be put into a queue for other processes.

        Returns:
            A Status object
        """

        assert self._gradient_descent_intervals >= 0, "Gradient descent intervals should be >= 0"
        self.init_main_loop()

        current_branch = None
        if start_branch.forced_input_bounds is None:
            start_branch.forced_input_bounds = self._bounds.forced_input_bounds

        self._branches.append(start_branch)

        while True:

            if finished_flag is not None and finished_flag.is_set():
                self._cleanup()
                return None

            if (needs_branches is not None and
                    put_queue is not None and
                    len(self._branches) >= queue_depth and
                    (self._branches[-1].depth - self._branches[0].depth) >= queue_depth and
                    needs_branches()):
                put_queue(self._branches.popleft())

            if len(self._branches) == 0:
                self._status = Status.Safe
                break

            new_branch = self._branches.pop()
            success = self._switch_branch(current_branch, new_branch)

            # ESIP failed, got invalid bounds
            if not success:
                self._status = Status.Underflow
                break
            current_branch = new_branch

            self.branches_explored += 1
            if current_branch is not None and current_branch.depth > self.max_depth:
                self.max_depth = current_branch.depth

            # Run LPSolver
            do_grad_descent = (False if self._gradient_descent_intervals == 0 else
                               (current_branch.depth % self._gradient_descent_intervals) == 0)
            self._status = self._verify_once(do_grad_descent, current_branch)

            if self._verbose:
                print(f" Depth: {current_branch.depth}, _status: {self._status}")

            # LPSolver found counterexample
            if self._status == Status.Unsafe:
                break

            # LPSolver returned safe
            if self._status == Status.Safe:
                continue

            if self._no_split:
                break

            did_branch = False
            if self._status == Status.Undecided:
                did_branch = self._branch(current_branch)

            if not did_branch:
                self._verification_objective.initial_settings(self._lp_solver, self._bounds,
                                                              current_branch.safe_classes)
                self._status = self._verify_once(True, current_branch)
                if self.status == Status.Undecided:
                    self._status = Status.Underflow
                break

        self._cleanup()
        return self._status

    def _verify_once(self, do_grad_descent: bool, current_branch: Branch) -> Status:

        """
        Run one step of the verification algorithm without refinement

        Args:
            do_grad_descent : If true gradient descent is done to find a counter example
            current_branch  : The current branch
        Returns:
            A Status object
        """

        self._counter_example = None

        if self._verification_objective.is_safe(self._bounds):
            return Status.Safe

        else:
            counter = 0
            while self._verification_objective.configure_next_potential_counter(self._lp_solver, self._bounds):
                result = self._lp_solver.solve()

                if not result:
                    self._verification_objective.finished_potential_counter(self._lp_solver, Status.Safe)
                    continue

                counter += 1
                # Get possible counter examples from LPSolver
                lp_counter_example, lp_output = self._lp_solver.get_assigned_values()
                lp_output = np.atleast_2d(lp_output)

                # Do gradient descent
                loss = self._verification_objective.grad_descent_losses(lp_output, self._bounds)

                self._counter_example = self._grad_descent_counter_example(lp_counter_example, loss,
                                                                           do_grad_descent)
                if self._counter_example is not None:
                    self._verification_objective.finished_potential_counter(self._lp_solver, Status.Undecided)
                    return Status.Unsafe

                self._verification_objective.finished_potential_counter(self._lp_solver, Status.Undecided)

            if counter == 0:
                return Status.Safe
            else:
                current_branch.safe_classes = self._verification_objective.safe_classes
                return Status.Undecided

    def init_main_loop(self):

        """
        Performs necessary initialization before running main verification loop
        """

        self.max_depth = 0
        self.branches_explored = 0
        self._init_bounds()

    # noinspection PyArgumentList,PyUnresolvedReferences
    def _grad_descent_counter_example(self, lp_counter_example: np.array, loss_func: Callable,
                                      do_grad_descent: bool = True) -> np.array:

        """
        Runs gradient descent updating the input to find true counter examples

        Args:
            lp_counter_example   : The counter example candidate from the LPSolver
            loss_func            : The loss function used for gradient descent
            do_grad_descent      : If true, gradient descent is performed to find a counter-example

        Returns:
            The counter example if found, else None.
        """

        x = torch.Tensor(lp_counter_example).clone().reshape(1, *self._verification_objective.input_shape)
        x.requires_grad = True
        y = self._get_logits(x)

        if self._verification_objective.is_counter_example(y.detach().numpy()):
            return x.detach().numpy()

        if not do_grad_descent:
            return None

        optimizer = optim.Adam([x], lr=self._gradient_descent_step, betas=(0.5, 0.9))

        old_loss = 1e10
        input_bounds = torch.Tensor(self._verification_objective.input_bounds_flat)

        for i in range(self._gradient_descent_max_iters):
            optimizer.zero_grad()
            loss = loss_func(y)
            loss.backward()
            optimizer.step()

            # Clip input to bounds
            x = x.view(-1)
            x.data[x < input_bounds[:, 0]] = input_bounds[:, 0][x < input_bounds[:, 0]]
            x.data[x > input_bounds[:, 1]] = input_bounds[:, 1][x > input_bounds[:, 1]]
            x = x.view(1, *self._verification_objective.input_shape)

            y = self._get_logits(x)
            if self._verification_objective.is_counter_example(y.detach().numpy()):
                return x.detach().numpy()

            if abs((loss - old_loss) / old_loss) < self._gradient_descent_min_loss_change:
                return None

            old_loss = loss

        return None

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:

        """
        Gets the logits from the model with input x

        Args:
            x:  The input vector to the neural network
        Returns:
            logits
        """

        try:
            self._model(x)
            logits = self._model.logits
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            return logits

        except AttributeError as e:
            raise LogitsException("Neural network has to implement attribute logits") from e

    def _branch(self, current_branch: Branch) -> bool:

        """
        Determines which node to split on and stores the branch data in self._branches

        Args:
            current_branch  : The current branch
        Returns:
            True if branching succeeded else false
        """

        refine_output_weights = self._verification_objective.output_refinement_weights(self._bounds)
        split = self.bounds.largest_error_split_node(output_weights=refine_output_weights)
        if split is None:
            return False
        layer, node = split

        lower = self._bounds.bounds_concrete[layer - 1][node][0]
        upper = self._bounds.bounds_concrete[layer - 1][node][1]
        split_x = self._bounds.mappings[layer].split_point(lower, upper)

        self._bounds.merge_current_bounds_into_forced()
        forced_input_bounds = self._bounds.forced_input_bounds

        # Add the lower split branch
        split_forced = [arr.copy() for arr in forced_input_bounds]
        old_forced = split_forced[layer - 1][node, 1]
        split_forced[layer - 1][node, 1] = split_x if split_x < old_forced else old_forced

        new_split = {"layer": layer, "node": node, "split_x": split_x, "upper": False}
        split_list = current_branch.split_list.copy()
        split_list.append(new_split)
        new_branch = Branch(current_branch.depth + 1, split_forced, split_list)
        new_branch.safe_classes = current_branch.safe_classes.copy()
        self._branches.append(new_branch)

        # Add the upper split branch
        split_forced = [arr.copy() for arr in forced_input_bounds]
        old_forced = split_forced[layer - 1][node, 0]
        split_forced[layer - 1][node, 0] = split_x if split_x > old_forced else old_forced

        new_split = {"layer": layer, "node": node, "split_x": split_x, "upper": True}
        split_list = current_branch.split_list.copy()
        split_list.append(new_split)
        new_branch = Branch(current_branch.depth + 1, split_forced, split_list)
        new_branch.safe_classes = current_branch.safe_classes.copy()
        self._branches.append(new_branch)

        return True

    def _switch_branch(self, current_branch: Branch, new_branch: Branch):

        """
        Switches from current branch to the new branch updating data in ESIP and LPSolver

        Args:
            current_branch  : The current branch
            new_branch      : The new branch

        Returns:
            False if update_nn_bounds() fails, du to invalid bounds (This can happen if the system is Safe).
        """

        assert new_branch is not None, "New branch was None"

        # Set forced bounds of ESIP
        self._bounds.forced_input_bounds = new_branch.forced_input_bounds

        # New split, recalculate affected layers
        if current_branch is not None and new_branch.depth == current_branch.depth + 1:

            success = self._bounds.calc_bounds(self._verification_objective.input_bounds_flat,
                                               from_layer=new_branch.split_list[-1]["layer"])

        # Backtracking, recalculate from first differing layer, but do not recalculate input bounds to first layer
        elif current_branch is not None and 0 < new_branch.depth <= current_branch.depth:

            min_layer = min([split["layer"] for split in current_branch.split_list[new_branch.depth - 1:]])
            success = self._bounds.calc_bounds(self._verification_objective.input_bounds_flat, from_layer=min_layer)

        # First call, calculate all bounds
        else:
            success = self._bounds.calc_bounds(self._verification_objective.input_bounds_flat, from_layer=1)

        if not success:
            return False

        self._configure_lp_solver()

        self._verification_objective.cleanup(self._lp_solver.grb_solver)
        self._verification_objective.initial_settings(self._lp_solver, self._bounds, new_branch.safe_classes)

        # Add branching constraints to LPSolver
        if (new_branch.depth > 0) and (current_branch is not None):
            new_branch.update_constrs(self._bounds, self._lp_solver, current_branch.split_list,
                                      current_branch.lp_solver_constraints)
        elif new_branch.depth > 0:
            # No current branch (new process), all constraints should be added
            new_branch.add_all_constrains(self._bounds, self._lp_solver, new_branch.split_list)

        return True

    def _cleanup(self):

        """
        Resets all stats specific to the last VerificationObjective
        """

        if self._lp_solver is not None:
            self._verification_objective.cleanup(self._lp_solver.grb_solver)
            self._lp_solver = None

        self._verification_objective = None
        self._bounds.reset_datastruct()
        self._branches = deque([])

    @staticmethod
    def _set_parameters_requires_grad(model: nn, requires_grad: bool = False):

        """
        Set the requires_grad attribute of all parameters in a torch neural network.

        Args:
            model           : The torch neural network
            requires_grad   : The requires_grad attribute that will be set
        """

        for param in model.parameters():
            param.requires_grad = requires_grad


class VeriNetException(Exception):
    pass


class LogitsException(VeriNetException):
    pass
