"""
This file contains the main class for verification.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import time

import multiprocessing as mp
from copy import deepcopy
import numpy as np
import torch.nn as nn

from src.algorithm.verinet_worker import VeriNetWorker
from src.algorithm.verinet_util import Status, Branch
from src.algorithm.verification_objectives import VerificationObjective
from src.util.logger import get_logger
from src.util.config import *


class VeriNet:

    """
    The VeriNet master class, responsible for starting verification, creating worker processes and delegating work
    """

    def __init__(self,
                 model: nn,
                 gradient_descent_max_iters: int = 5,
                 gradient_descent_step: float = 1e-1,
                 gradient_descent_min_loss_change: float = 1e-2,
                 max_procs: int = None,
                 queue_depth: int = 10):

        """
        Args:
            model                           : The torch neural network, the requires_grad parameters of this model
                                              might be changed.
            gradient_descent_max_iters      : The number of iterations used in gradient descent to find a true counter
                                              example from a _lp_solver counter example
            gradient_descent_step           : The step size of the gradient descent used to find a true counter example
                                              from a _lp_solver counter example
            gradient_descent_min_loss_change: The minimum amount of change in loss from last iteration to keep trying
                                              gradient descent
            max_procs                       : The maximum number of processes, if None it is set to 2*cpu_count()
            queue_depth                     : If the depth difference between a branch and the deepest branch is more
                                              than this, the branch will be put into a queue for other processes.
        """

        self._model_nn = model

        self._gradient_descent_max_iters = gradient_descent_max_iters
        self._gradient_descent_step = gradient_descent_step
        self._gradient_descent_min_loss_change = gradient_descent_min_loss_change
        self._max_procs = mp.cpu_count() if max_procs is None else max_procs
        self._queue_depth = queue_depth

        self._gradient_descent_intervals = None
        self._timeout = None
        self._no_split = None
        self._verbose = None

        self._max_depth = mp.Value("i", 0)
        self._branches_explored = mp.Value("i", 0)
        self._status = mp.Value("i", Status.Undecided.value)
        self._counter_example = None
        self._verification_objective = None
        self._workers = []
        self._active_tasks = mp.Value("i", 0)
        self._active_procs = mp.Value("i", 0)
        self._work_lock = mp.Lock()
        self._active_tasks_lock = mp.Lock()
        self._branch_queue = mp.Manager().Queue()

        self._worker_join_timeout = 120

        self._finished_flag = mp.Event()
        self._all_children_done = mp.Event()

        self.logger = get_logger(LOGS_LEVEL, __name__, "../../logs/", "verinet_log")

    @property
    def max_depth(self):
        return self._max_depth.value

    @property
    def branches_explored(self):
        return self._branches_explored.value

    @property
    def status(self):
        return Status(self._status.value)

    @property
    def counter_example(self):
        return self._counter_example

    def verify(self,
               verification_objective: VerificationObjective,
               gradient_descent_intervals: int = 5,
               timeout: float = 3600,
               no_split: bool = False,
               verbose=True):

        """
        Starts the verification process

        Args:
            verification_objective          : The VerificationObjective
            no_split                        : If true no splitting is done
            timeout                         : The maximum time the process will run before timeout
            verbose                         : If true information is printed at each branch
            gradient_descent_intervals      : Gradient descent performed to find counter example each number of
                                              intervals. Should be >0 0.
        """

        start_time = time.time()

        self._reset_params()

        self._counter_example = mp.Array("f", np.zeros(verification_objective.input_size, dtype=np.float32))
        self._verification_objective = verification_objective
        self._gradient_descent_intervals = gradient_descent_intervals
        self._timeout = timeout
        self._no_split = no_split
        self._verbose = verbose

        # Try a one-shot verification before initializing children avoids overhead of multiprocessing and jit compiling
        self._one_shot_approximation()

        if self.status != Status.Undecided:
            return self._status

        self._reset_mp_params()

        branch = Branch(0, None, [])
        self._put_branch(branch)  # Add initial branch
        self.logger.debug(f"Main process put first branch {branch} on queue")

        self._start_workers()

        self.logger.debug("Main process waiting for workers")
        timeout = not self._finished_flag.wait(timeout=self._timeout - (time.time() - start_time))
        self.logger.debug(f"Main process finished waiting, timeout={timeout}")

        self._put_poison_pills()

        self._join_workers()

        if (not timeout) and (self._status.value is Status.Undecided.value):
            assert self._active_tasks.value == 0, "Ended before timeout without finishing all active tasks"
            self._status.value = Status.Safe.value

        self.logger.debug(f"Main process finished with status: {self.status}")
        return self.status

    def _one_shot_approximation(self):

        """
        Runs the algorithm once without branching

        Used by main process before starting children, to see if the system is solvable without overhead for
        multiprocessing or jit-compiling.
        """

        self.logger.debug("Starting one-shot approximation")
        solver = VeriNetWorker(self._model_nn,
                               verification_objective=deepcopy(self._verification_objective),
                               no_split=True,
                               gradient_descent_intervals=self._gradient_descent_intervals,
                               gradient_descent_max_iters=self._gradient_descent_max_iters,
                               gradient_descent_step=self._gradient_descent_step,
                               gradient_descent_min_loss_change=self._gradient_descent_min_loss_change,
                               verbose=self._verbose
                               )

        solver.verify(Branch(0, None, []), None, None, None, queue_depth=-1)

        if solver.status != Status.Undecided:
            self._status = solver.status
            self._max_depth.value = solver.max_depth
            self._branches_explored.value = solver.branches_explored
            self._counter_example = solver.counter_example

    def _start_workers(self):

        """
        Starts all workers
        """

        self.logger.debug("Starting workers")

        for i in range(self._max_procs):
            worker = mp.Process(target=self._start_worker)
            worker.start()
            self._workers.append(worker)
            self._active_procs.value += 1
            self.logger.debug(f"Added worker {i + 1} of {self._max_procs}")

    def _put_poison_pills(self):

        """
        Puts poison pills (None) into the queue to signal workers to quit.
        """

        self.logger.debug(f"Putting poison pills")
        with self._work_lock:
            self._finished_flag.set()
        for i in range(self._max_procs):
            self.logger.debug(f"Added poison pill {i + 1} of {self._max_procs}")
            self._branch_queue.put(None)  # "Poison pill for killing child processes"

    def _join_workers(self):

        """
        Joins all workers, if join fails after self._worker_join_timeout, the process is terminated
        """

        self._all_children_done.wait()  # Wait to avoid deadlock in self._branches_queue

        self.logger.debug(f"Main process joining workers")
        for worker in self._workers:
            worker.join(self._worker_join_timeout)
            if worker.exitcode is None:
                self.logger.warning(f"Main process could not join with worker, terminating instead")
                worker.terminate()

    def _start_worker(self):

        """
        Starts a worker process
        """

        while True:
            solver = VeriNetWorker(self._model_nn,
                                   verification_objective=deepcopy(self._verification_objective),
                                   no_split=self._no_split,
                                   gradient_descent_intervals=self._gradient_descent_intervals,
                                   gradient_descent_max_iters=self._gradient_descent_max_iters,
                                   gradient_descent_step=self._gradient_descent_step,
                                   gradient_descent_min_loss_change=self._gradient_descent_min_loss_change,
                                   verbose=self._verbose
                                   )

            branch = self._branch_queue.get()
            self.logger.debug(f"Worker retrieved branch {branch} from queue")
            with self._work_lock:
                if branch is None:
                    self._active_procs.value -= 1
                    if self._active_procs.value == 0:
                        self._all_children_done.set()
                    self.logger.debug(f"Worker exited")
                    return

            if self._finished_flag.is_set():
                # We have to empty self._branch_queue to avoid deadlock, can't return yet...
                continue

            solver.verify(branch, self._finished_flag, needs_branches=self._needs_branches,
                          put_queue=self._put_branch, queue_depth=self._queue_depth)

            self._finished_subtree(solver.max_depth, solver.branches_explored, solver.status, solver.counter_example)

    def _finished_subtree(self, max_depth: int, branches_explored: int,
                          status: Status, counter_example: np.array = None):

        """
        Called from workers when they finish their current subtree.

        This function updates the relevant statistics, such as max_depth and branches explored. If a worker finds
        the system to be unsafe or encounters underflow, the found solution flag is set.

        Args:
            max_depth           : The workers maximum branch depth
            branches_explored   : The number of branches the worker explored
            status              : The Status Enum with the final status of the subtree
            counter_example     : The counter example, if found.
        """

        with self._work_lock:

            self._max_depth.value = max_depth if max_depth > self._max_depth.value else self._max_depth.value
            self._branches_explored.value += branches_explored

            if not self._finished_flag.is_set() and status.value == Status.Unsafe.value:
                self._status.value = status.value
                self._counter_example = counter_example
                self._finished_flag.set()

            elif not self._finished_flag.is_set() and status.value == Status.Underflow.value:
                self._status.value = status.value
                self._finished_flag.set()

        with self._active_tasks_lock:
            self._active_tasks.value -= 1
            if self._active_tasks.value == 0:
                self._finished_flag.set()

    def _needs_branches(self) -> bool:

        """
        Returns true if the queue needs more branches

        If the current number of active tasks (running procs + queue size) is smaller than 2*max_procs this
        returns true
        """

        return self._active_tasks.value < (1.2 * self._max_procs)

    def _put_branch(self, branch: Branch):

        """
        This method is called by workers to put a branch into the queue.

        Args:
            branch  : The new branch
        """

        with self._active_tasks_lock:
            self._active_tasks.value += 1

        with self._work_lock:
            if not self._finished_flag.is_set():
                self.logger.debug(f"Worker put branch {branch} on queue")
                self._branch_queue.put(branch)

    def _reset_params(self):

        """
        Resets all object parameters.

        This makes the objective ready for the next verification task
        """

        self._max_depth = mp.Value("i", 0)
        self._branches_explored = mp.Value("i", 0)
        self._status = mp.Value("i", Status.Undecided.value)
        self._counter_example = None
        self._verification_objective = None

    def _reset_mp_params(self):

        """
        Resets all params used for multiprocessing
        """

        self._workers = []
        self._active_tasks = mp.Value("i", 0)
        self._active_procs = mp.Value("i", 0)
        self._work_lock = mp.Lock()
        self._active_tasks_lock = mp.Lock()
        self._branch_queue = mp.Manager().Queue()
        self._finished_flag = mp.Event()
        self._all_children_done = mp.Event()
