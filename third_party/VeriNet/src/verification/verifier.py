"""
The VeriNet verification toolkit for neural networks.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

# noinspection PyUnresolvedReferences
import os
import time
from typing import Callable, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from verinet.verification.objective import Objective
from verinet.verification.lp_solver import LPSolver, LPSolverException
from verinet.verification.verifier_util import Status, Branch
from verinet.sip_torch.ssip import SSIP
from verinet.sip_torch.rsip import RSIP
from verinet.util.logger import get_logger
from verinet.util.config import CONFIG

logger = get_logger(CONFIG.LOGS_LEVEL_VERIFIER, __name__, "../../logs/", "verifier_log")


class Verifier:

    def __init__(self,
                 model: nn,
                 objective: Objective,
                 use_progress_bars: bool = False):

        """
        Args:
            model:
                The torch neural network, the requires_grad parameters of this _model
                might be changed.
            objective:
                The verification Objective.
            use_progress_bars:
                Determines whether to use progress bars.
        """

        self._model = model
        self._objective = objective

        self._pbar = None
        self._pbar_last_update_time = None
        self._use_progress_bars = use_progress_bars

        self._gradient_descent_intervals = CONFIG.GRADIENT_DESCENT_INTERVAL
        self._gradient_descent_max_iters = CONFIG.GRADIENT_DESCENT_MAX_ITERS
        self._gradient_descent_step = CONFIG.GRADIENT_DESCENT_STEP
        self._gradient_descent_min_loss_change = CONFIG.GRADIENT_DESCENT_MIN_LOSS_CHANGE

        self._max_est_memory_usage = CONFIG.MAX_ESTIMATED_MEM_USAGE
        self._max_queued_branches = CONFIG.MAX_QUEUED_BRANCHES

        self._status = Status.Undecided
        self._counter_example = None

        self._lp_solver = None
        self._rsip = None

        # If not None, bounds will be recalculated at least from this node on the next call to _recalculate_bounds()
        self._recalculate_from_node = None

        self._branches = deque([])

        self._set_parameters_requires_grad(model=model, requires_grad=False)

        self.max_depth = 0
        self.branches_explored = 0

        self._potential_cex = []

        self._init_sip()

    @property
    def counter_example(self) -> np.array:
        return self._counter_example

    @property
    def status(self) -> Status:
        return self._status

    @property
    def branches(self) -> deque:
        return self._branches

    def init_main_loop(self):

        """
        Performs necessary initialization before running main verification loop.
        """

        self.reset_params()
        self.max_depth = 0
        self.branches_explored = 0
        self._recalculate_from_node = 0
        self._init_sip()

    def reset_params(self):

        """
        Resets all params specific to the last run of verify().
        """

        if self._lp_solver is not None:
            self._objective.cleanup(self._lp_solver)
            self._lp_solver.remove_all_constraints()

        self._status = Status.Undecided
        self._counter_example = None
        self._rsip = None
        self._branches = deque([])

        self.max_depth = 0
        self.branches_explored = 0

    # noinspection PyArgumentList,PyCallingNonCallable
    def _init_sip(self):

        """
        Initializes the SIP object
        """

        self._rsip = RSIP(self._model, torch.LongTensor(self._objective.input_shape),
                          max_est_memory_usage=self._max_est_memory_usage, use_pbar=self._use_progress_bars)
        self._ssip = SSIP(self._model, torch.LongTensor(self._objective.input_shape))

    def _configure_lp_solver(self):

        """
        Initializes the LPSolver and sets the input bounds.
        """

        self._lp_solver = LPSolver(self._objective.input_size)
        self._lp_solver.set_input_bounds(self._objective.input_bounds_flat)

    def _update_pbar(self, timeout: float):

        """
        Updates the progress bar if self._use_progress_bars is True.

        Args:
            timeout:
                The timeout parameter of the verification loop
        """

        if not self._use_progress_bars:
            return

        if self._pbar is None:
            self._pbar = tqdm(total=timeout, leave=False, desc="Main verification loop (to timeout)")
            self._pbar_last_update_time = time.time()

        else:
            new_time = time.time()
            self._pbar.update(int(new_time - self._pbar_last_update_time))
            self._pbar_last_update_time = new_time

    # noinspection PyUnboundLocalVariable
    def verify(self,
               start_branch: Branch = None,
               timeout: float = None,
               no_split: bool = False,
               finished_flag: Optional[mp.Event] = None,
               needs_branches: Optional[Callable] = None,
               put_queue: Optional[Callable] = None,
               queue_depth: int = None) -> Optional[Status]:

        """
        Runs the main VeriNet for verification of the given verification
        objective.

        Args:
            start_branch:
                The first branch.
            timeout:
                The timeout parameter. If None then timeout is disabled.
            no_split:
                If true, no splitting is done.
            finished_flag:
                If set, the loop will abort. If None this parameter is disregarded.
            needs_branches:
                A callable that should return true if a call to pu_queue() is required.
                Used in multiprocessing to determine when the shared queue needs new
                branches. If None, put_queue is never called.
            put_queue:
                A callable to put new branches into the queue. Use in multiprocessing
                to queue branches in a shared queue. If None, branches will never be
                added to queue.
            queue_depth:
                The put_queue callable is only called if the depth difference between
                the shallowest branch and the deepest branch is more than this number.

        Returns:
            The answer to the verification query as a Status object.
        """

        start_time = time.time()
        self._lp_solver = None

        assert self._gradient_descent_intervals >= 0, "Gradient descent intervals should be >= 0"
        self.init_main_loop()

        self._recalculate_from_node = 0

        current_branch = None
        if start_branch is None:
            start_branch = Branch(0, [None for _ in range(self._rsip.num_nodes)], [])
        if start_branch.forced_bounds_pre is None:
            start_branch.forced_bounds_pre = [None for _ in range(self._rsip.num_nodes)]

        self._branches.append(start_branch)

        while True:

            self._update_pbar(timeout)

            if finished_flag is not None and finished_flag.is_set():
                return None

            if (timeout is not None and (time.time() - start_time) > timeout) or \
                    (self._max_queued_branches is not None and (len(self._branches) > self._max_queued_branches)):
                self._status = Status.Undecided

                self._close_pbar()

                return Status.Undecided

            # Put branches on multiprocessing queue
            if (needs_branches is not None and put_queue is not None and len(self._branches) >= queue_depth and
                    (self._branches[-1].depth - self._branches[0].depth) >= queue_depth and needs_branches()):
                put_queue(self._branches.popleft())

            # No more branches, problem is safe
            if len(self._branches) == 0:
                self._status = Status.Safe
                break

            new_branch = self._branches.pop()
            self.branches_explored += 1

            success = self._switch_branch(current_branch, new_branch)

            if not success:  # Branch constraints were unsatisfiable in SIP
                self._status = Status.Safe
                continue
            current_branch = new_branch

            if current_branch is not None and current_branch.depth > self.max_depth:
                self.max_depth = current_branch.depth

            # Run the solver
            do_grad_descent = (False if self._gradient_descent_intervals == 0 else
                               (current_branch.depth % self._gradient_descent_intervals) == 0)
            self._status = self._verify_once(do_grad_descent, current_branch)

            if self._status == Status.Unsafe or self._status == Status.Underflow:
                break

            if self._status == Status.Safe:
                continue

            if no_split:
                if put_queue is not None:
                    # Put remaining branches on multiprocessing queue
                    did_branch = self._branch(current_branch)
                    if not did_branch:
                        self._status = Status.Underflow
                        break

                    while len(self._branches) > 0:
                        put_queue(self._branches.popleft())
                break

            did_branch = False
            if self._status == Status.Undecided:
                did_branch = self._branch(current_branch)

            if not did_branch:
                self._status = self._verify_once(True, current_branch)
                if self.status == Status.Undecided:
                    self._status = Status.Underflow
                break

        self._close_pbar()

        return self._status

    def pre_process_attack(self):

        """
        Performs a simple adversarial attack in an attempt to find a counter example.

        Note that this method should never be called within the main verification
        loop in verify(), it is meant for external use only.
        """

        counter_example = None
        device = self._model.device

        for constr_idx in range(0, self._objective.num_constraints):

            self._objective.current_constraint_idx = constr_idx
            self._rsip.nodes[0].bounds_concrete_pre = \
                [self._objective.input_bounds_flat_torch.clone().detach().to(device)]

            mid_point = (self._objective.input_bounds_flat_torch[:, 1] +
                         self._objective.input_bounds_flat_torch[:, 0]).to(device=device) / 2
            mid_point = mid_point.double() if self._model.uses_64bit else mid_point.float()

            loss_func = self._objective.grad_descent_loss

            counter_example = self._grad_descent_counter_example(potential_cex=mid_point,
                                                                 loss_func=loss_func,
                                                                 do_grad_descent=True)

            if counter_example is not None:
                self._counter_example = counter_example
                self._status = Status.Unsafe
                break

        self._objective.current_constraint_idx = None
        del self._rsip.nodes[0].bounds_concrete_pre

        torch.cuda.empty_cache()

        return counter_example

    def _close_pbar(self):

        """
        Closes the pbar.
        """

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    # noinspection PyTypeChecker
    def _verify_once(self, do_grad_descent: bool, current_branch: Branch) -> Status:

        """
        Run one step of the verification.

        Args:
            do_grad_descent:
                If true gradient descent is done to find a counter example.
            current_branch:
                The current branch.
        Returns:
            A Status object.
        """

        self._counter_example = None
        self._objective.safe_constraints = current_branch.safe_constraints
        self._potential_cex = []

        while True:

            try:
                finished, potential_cex, max_val = self._objective.find_potential_cex(current_branch,
                                                                                      self._lp_solver,
                                                                                      self._rsip)
            except LPSolverException:
                return Status.Underflow

            if finished:
                break

            self._potential_cex.append((self._objective.current_constraint_idx, max_val, potential_cex))

            if potential_cex is None:
                self._objective.finished_constraint(self._lp_solver, Status.Safe)
                continue

            loss = self._objective.grad_descent_loss
            self._counter_example = self._grad_descent_counter_example(potential_cex, loss, do_grad_descent)
            self._objective.finished_constraint(self._lp_solver, Status.Undecided)

            if self._counter_example is not None:
                return Status.Unsafe

        if len([cex for cex in self._potential_cex if cex[2] is not None]) == 0:
            return Status.Safe
        else:
            current_branch.safe_constraints = self._objective.safe_constraints
            return Status.Undecided

    # noinspection PyArgumentList,PyUnresolvedReferences
    def _grad_descent_counter_example(self, potential_cex: np.array, loss_func: Callable,
                                      do_grad_descent: bool = True) -> np.array:

        """
        Runs gradient descent updating the input to find true counter examples.

        Args:
            potential_cex:
                The counter example candidate.
            loss_func:
                The loss function used for gradient descent.
            do_grad_descent:
                If true, gradient descent is performed to find a counter-example.

        Returns:
            The counter example if found, else None.
        """

        x = potential_cex.view(1, *self._objective.input_shape).to(device=self._model.device).clone()

        if self._model.uses_64bit:
            input_bounds = self._rsip.get_bounds_concrete_pre(0)[0].double()
        else:
            input_bounds = self._rsip.get_bounds_concrete_pre(0)[0].float()

        # Clip input to bounds
        x = x.view(-1)
        x.data[x < input_bounds[:, 0]] = input_bounds[:, 0][x < input_bounds[:, 0]]
        x.data[x > input_bounds[:, 1]] = input_bounds[:, 1][x > input_bounds[:, 1]]
        x = x.view(1, *self._objective.input_shape).detach().clone()

        x.requires_grad = True

        self._model.eval()
        y = self._model(x)[0]

        if self._objective.is_counter_example(y):
            return x.cpu().detach().numpy()

        if not do_grad_descent:
            return None

        optimizer = optim.Adam([x], lr=self._gradient_descent_step, betas=(0.5, 0.9))

        old_loss = 1e10

        for i in range(self._gradient_descent_max_iters):
            optimizer.zero_grad()
            loss = loss_func(y)
            loss.backward()
            optimizer.step()

            # Clip input to bounds
            x = x.view(-1)
            x.data[x < input_bounds[:, 0]] = input_bounds[:, 0][x < input_bounds[:, 0]]
            x.data[x > input_bounds[:, 1]] = input_bounds[:, 1][x > input_bounds[:, 1]]
            x = x.view(1, *self._objective.input_shape)

            y = self._model(x)[0]

            if self._objective.is_counter_example(y):
                return x.cpu().detach().numpy()

            if ((old_loss - loss) / old_loss) < self._gradient_descent_min_loss_change:
                return None

            old_loss = loss

        return None

    # noinspection PyCallingNonCallable
    def _branch(self, current_branch: Branch) -> bool:

        """
        Determines which neuron to split on and stores the branch data in self._branches.

        Args:
            current_branch:
                The current branch.
        Returns:
            True if branching succeeded else false.
        """

        if CONFIG.USE_OPTIMISED_RELAXATION_SPLIT_HEURISTIC:

            # Optimise relaxations for largest violating cex.

            potential_cex = [pot_cex for pot_cex in self._potential_cex if pot_cex[2] is not None]
            potential_cex = sorted(potential_cex, key=lambda x: x[1])[0]  # Sort after largest output value.

            self._model.forward(potential_cex[2].view((1, *self._objective.input_shape)), cleanup=False)
            self._objective.calc_optimised_relaxations(self._rsip)

            for node in self._model.nodes:
                node.value = None
                node.input_value = None

            self._rsip.set_optimised_relaxations()

        output_eq = torch.FloatTensor(self._objective.get_summed_constraints())

        impact = self._rsip.get_most_impactfull_neurons(output_eq, lower=False)
        self._rsip.set_non_parallel_relaxations()

        if len(impact[1]) == 0:
            return False
        else:
            node, neuron = impact[1][0]

        node, neuron = int(node), int(neuron)

        lower, upper = self._rsip.get_bounds_concrete_pre(node)[0][neuron].cpu()
        split_x = self._rsip.get_split_point(lower, upper, node)

        self._branches.append(self._new_branch(node, neuron, split_x, current_branch, upper=True))
        self._branches.append(self._new_branch(node, neuron, split_x, current_branch, upper=False))

        return True

    def _new_branch(self, node: int, neuron: int, split_x: float, current_branch: Branch, upper: bool = False):

        """
        Creates a new Branch objects from the given parameters.

        Args:
            node:
                The node number
            neuron:
                The neuron number
            split_x:
                The x value where to split the activation function
            current_branch:
                The current branch object
            upper:
                If true, the upper branch (x > split_x) is returned, else the lower is
                returned.
        Return:
            The new branch object.
        """

        split_forced = [bounds if bounds is not None else None for bounds in self._rsip.get_forced_bounds_pre(True)]

        if len(split_forced[node]) != 1:
            raise ValueError(f"Expected one set of forced bounds for split node, got {len(split_forced[node])}")

        if not upper:
            old_forced = split_forced[node][0][neuron, 1]
            split_forced[node][0][neuron, 1] = split_x if split_x < old_forced else old_forced
        else:
            old_forced = split_forced[node][0][neuron, 0]
            split_forced[node][0][neuron, 0] = split_x if split_x > old_forced else old_forced

        new_split = {"node": node, "neuron": neuron, "split_x": split_x, "upper": upper}
        split_list = current_branch.split_list.copy()
        split_list.append(new_split)
        new_branch = Branch(current_branch.depth + 1, split_forced, split_list)
        new_branch.safe_constraints = current_branch.safe_constraints.copy()

        return new_branch

    def _switch_branch(self, current_branch: Branch, new_branch: Branch):

        """
        Switches from current branch to the new branch updating data in SIP.

        Args:
            current_branch:
                The current branch.
            new_branch:
                The new branch.
        Returns:
            False if update_nn_bounds() fails, du to invalid bounds (This can happen if
            the system is Safe).
        """

        assert new_branch is not None, "New branch was None"

        readd_all_constraints = self._recalculate_from_node is not None

        if not self._recalculate_bounds(current_branch, new_branch):
            return False

        if self._lp_solver is None:
            self._configure_lp_solver()

        if CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS \
                and self._lp_solver.num_bias_vars < self._rsip.num_non_linear_neurons:
            self._lp_solver.add_bias_variables(self._rsip.num_non_linear_neurons - self._lp_solver.num_bias_vars)

        self._objective.cleanup(self._lp_solver)

        # Add branching constraints to LPSolver
        if readd_all_constraints:
            self._lp_solver.remove_all_constraints()
            new_branch.add_all_constrains(self._rsip, self._lp_solver, new_branch.split_list)
        elif (new_branch.depth > 0) and (current_branch is not None):
            new_branch.update_constraints(self._rsip, self._lp_solver, current_branch.split_list,
                                          current_branch.lp_solver_constraints)
        elif new_branch.depth > 0:
            # No current branch (new process), all constraints should be added
            new_branch.add_all_constrains(self._rsip, self._lp_solver, new_branch.split_list)

        return True

    def _recalculate_bounds(self, current_branch: Branch, new_branch: Branch):

        """
        Updates the bounds in SIP for the new branch.
        Args:
            current_branch:
                The current branch.
            new_branch:
                The new branch.
        Returns:
            False if update_nn_bounds() fails, due to invalid bounds (This can happen if
            the system is Safe).
        """

        # Set forced bounds of SIP
        self._rsip.set_forced_bounds_pre(new_branch.forced_bounds_pre)

        input_bounds = self._objective.input_bounds_flat_torch

        # New split, recalculate affected nodes
        if current_branch is not None and new_branch.depth == current_branch.depth + 1:

            from_node = int(new_branch.split_list[-1]["node"])

            if self._recalculate_from_node is not None:
                from_node = min(from_node, self._recalculate_from_node)

            if from_node != 0:
                self._rsip.update_modified_neurons(from_node)
                from_node += 1

            success = self._calc_bounds(input_bounds, from_node)
            self._recalculate_from_node = None if success else from_node

        # Backtracking, recalculate from first differing node, but do not recalculate input bounds to first node
        elif current_branch is not None and 0 < new_branch.depth <= current_branch.depth:

            from_node = int(min([new_branch.split_list[-1]["node"]] +
                                [split["node"] for split in current_branch.split_list[new_branch.depth - 1:]]))

            if self._recalculate_from_node is not None:
                from_node = int(min(from_node, self._recalculate_from_node))

            if from_node != 0:
                self._rsip.update_modified_neurons(from_node)
                from_node += 1

            success = self._calc_bounds(input_bounds, from_node)

            self._recalculate_from_node = None if success else from_node

        # First call, calculate all bounds
        else:
            self._recalculate_from_node = None
            success = self._calc_bounds(input_bounds, from_node=0)

        return success

    # noinspection PyArgumentList,PyCallingNonCallable
    def _calc_bounds(self, input_bounds: np.array, from_node: int):

        """
        Calculates bounds using sip.

        Args:
            input_bounds:
                The concrete lower and upper input-bounds.
            from_node:
                The node from where ESIP should be recalculated.
        Returns:
            True if the current configuration lead to valid bounds.
        """

        if CONFIG.USE_SSIP:
            if not CONFIG.STORE_SSIP_BOUNDS:
                ssip = SSIP(self._model, input_shape=torch.LongTensor(self._objective.input_shape))
                ssip.set_forced_bounds_pre(self._rsip.get_forced_bounds_pre())
                ssip.calc_bounds(input_bounds)
                ssip.merge_current_bounds_into_forced()
                self._rsip.set_forced_bounds_pre(ssip.get_forced_bounds_pre())
                del ssip

            else:
                self._ssip.set_forced_bounds_pre(self._rsip.get_forced_bounds_pre())
                self._ssip.calc_bounds(input_bounds, from_node=from_node)
                self._ssip.merge_current_bounds_into_forced()
                self._rsip.set_forced_bounds_pre(self._ssip.get_forced_bounds_pre())

        success = self._rsip.calc_bounds(input_bounds, from_node=from_node)
        self._rsip.merge_current_bounds_into_forced()
        self._rsip.max_non_linear_rate = 1
        self._rsip.max_non_linear_rate_split_nodes = 1

        return success

    @staticmethod
    def _set_parameters_requires_grad(model: nn, requires_grad: bool = False):

        """
        Set the requires_grad attribute of all parameters in a torch neural network.

        Args:
            model:
                The torch neural network.
            requires_grad:
                The requires_grad attribute that will be set.
        """

        for param in model.parameters():
            param.requires_grad = requires_grad


class VeriNetException(Exception):
    pass


class LogitsException(VeriNetException):
    pass
