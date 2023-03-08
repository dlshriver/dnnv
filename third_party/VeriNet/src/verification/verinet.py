"""
This file contains the main class for verification.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import pickle
import time
import psutil

import torch
import torch.multiprocessing as mp

torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')  # Reduces the number of file descriptors opened by torch

from multiprocessing.managers import BaseManager

import numpy as np

from verinet.verification.verifier import Verifier
from verinet.verification.verifier_util import Status, Branch
from verinet.verification.objective import Objective
from verinet.util.logger import get_logger
from verinet.util.config import CONFIG

logger = get_logger(CONFIG.LOGS_LEVEL_VERINET, __name__, "../../logs/", "verinet_log")


class ParameterManager:

    """
    Manager class used for variables shared between processes.
    """

    def __init__(self):
        self._objective = None
        self._counter_example = None

    def get_objective(self):
        return self._objective

    def set_objective(self, objective: Objective):
        self._objective = objective

    def get_counter_example(self):
        return self._counter_example

    def set_counter_example(self, counter_example: np.array):
        self._counter_example = counter_example


class CustomManager(BaseManager):
    pass


CustomManager.register('ParameterManager', ParameterManager)

custom_manager = CustomManager()
custom_manager.start()


class VeriNet:
    """
    The VeriNet master class, responsible for starting verification, creating worker
    processes and delegating work.
    """

    # noinspection PyUnresolvedReferences,PyDictCreation
    def __init__(self, max_procs: int = None, use_gpu: bool = False):

        self._mp_context = self._set_mp_context()

        self._max_procs = self._mp_context.cpu_count() if max_procs is None else max_procs
        self._parameter_manager = custom_manager.ParameterManager()

        # Variables used in verify()
        self._timeout = None

        # Solution and stats variables
        self.max_depth = None
        self.branches_explored = None
        self.status = None
        self.counter_example = None

        # Multiprocessing solution and stats variables
        self._max_depth_mp = self._mp_context.Value("i", 0)
        self._branches_explored_mp = self._mp_context.Value("i", 0)
        self._status_mp = self._mp_context.Value("i", Status.Undecided.value)

        self._queue_depth = CONFIG.QUEUE_DEPTH

        # Multiprocessing variables
        self._workers = None
        self._active_tasks = self._mp_context.Value("i", 0)
        self._active_workers = self._mp_context.Value("i", 0)
        self._branch_queue = self._mp_context.Manager().Queue()
        self._work_lock = self._mp_context.Lock()
        self._active_tasks_lock = self._mp_context.Lock()

        # Multiprocess flags
        self._finished_flag = self._mp_context.Event()
        self._all_children_done = self._mp_context.Event()
        self._destroy_flag = self._mp_context.Event()

        self._activate_master_worker = self._mp_context.Event()
        self._activate_slave_workers = self._mp_context.Event()

        self._main_pid = os.getpid()
        self._master_worker_pid = self._mp_context.Value("i", 0)
        self._master_worker_should_use_gpu = use_gpu
        self._initial_mem_usage = None

        self._init_workers()

        self._handle_memory()

    # noinspection PyArgumentList
    def __del__(self):

        """
        Last resort attempt for cleaning up stuck resource tracker processes.
        """

        try:
            self.cleanup()
        except TypeError:
            pass

    @property
    def objective(self):
        return self._parameter_manager.get_objective()

    @objective.setter
    def objective(self, val):
        self._parameter_manager.set_objective(val)

    @property
    def _max_depth(self):
        return self._max_depth_mp.value

    @_max_depth.setter
    def _max_depth(self, val):
        self._max_depth_mp.value = val

    @property
    def _branches_explored(self):
        return self._branches_explored_mp.value

    @_branches_explored.setter
    def _branches_explored(self, val):
        self._branches_explored_mp.value = val

    @property
    def _status(self):
        return Status(self._status_mp.value)

    @_status.setter
    def _status(self, status):
        self._status_mp.value = status.value

    @property
    def _counter_example(self):
        return self._parameter_manager.get_counter_example()

    @_counter_example.setter
    def _counter_example(self, counter_example):
        self._parameter_manager.set_counter_example(counter_example)

    @property
    def _num_workers(self):
        if self._workers is None:
            return 0
        else:
            return len(self._workers)

    @staticmethod
    def _set_mp_context():

        """
        Decides whether to use fork() or spawn() depending on if Cuda is enabled.

        Notice that spawn() is significantly slower than fork(). However, fork() fails
        if Cuda is enabled.
        """

        if CONFIG.CHILDPROC_TYPE == "fork":
            if torch.cuda.is_available():  # Fork() only works if cuda is disabled.
                logger.warning("Fork is not supported with Cuda enabled, switching to spawn.\n")
                return mp.get_context("spawn")
            else:
                return mp.get_context("fork")

        elif CONFIG.CHILDPROC_TYPE == "spawn":
            return mp.get_context("spawn")

        elif CONFIG.CHILDPROC_TYPE == "forkserver":
            ctx = mp.get_context("forkserver")
            ctx.set_forkserver_preload(['.verifier', '.objective', 'multiprocessing.managers', 'pickle', 'Objective',
                                        'Verifier', 'BaseManager'])
            return ctx

        else:
            raise ValueError(f"config.childproc_type: {CONFIG.CHILDPROC_TYPE} not recognised")

    def _init_workers(self):

        """
        Initialises workers
        """

        self._all_children_done.clear()

        if self._workers is None or self._num_workers < self._max_procs:

            self._create_workers(self._max_procs - self._num_workers)
            # noinspection PyUnresolvedReferences
            self._master_worker_pid.value = self._workers[0].pid

        for i in range(self._active_workers.value):
            with self._active_tasks_lock:
                self._active_tasks.value += 1

            logger.debug(f"Added poison pill {i + 1} of {self._max_procs}")
            self._branch_queue.put(None)  # "Poison pill for killing child processes"

    def verify(self,
               objective: Objective,
               timeout: float = None):

        """
        Starts the verification process.

        Args:
            objective:
                The verification Objective.
            timeout:
                The maximum time the process will run before timeout.
        """

        start_time = time.time()

        self._reset_params()

        if CONFIG.USE_ONE_SHOT_ATTEMPT:
            status = self.one_shot_attempt(objective)
            if status == Status.Safe or status == Status.Unsafe:
                return status
        else:
            self._all_children_done.wait()
            self._put_branch(Branch(0, None, []))

        # Calculated values retain the computational graph making the model unpickable, thus failing multiprocessing.
        for node in objective.model.nodes:
            if hasattr(node, "value"):
                del node.value

        # Pickling objective before storing as a shared variable speeds up the process
        p = pickle.dumps(objective)
        self.objective = p

        self._all_children_done.clear()

        self._set_activate_master_worker_flags()
        self._set_activate_slave_workers_flags()
        did_timeout = self._wait_for_workers(timeout - (time.time() - start_time))

        self._suspend_workers()

        if (not did_timeout) and (self._status is Status.Undecided):
            assert self._active_tasks.value == 0, "Ended before timeout without finishing all active tasks"
            self._status = Status.Safe

        logger.debug(f"Main process finished with status: {self._status}")

        self._handle_memory()

        if not self._status == Status.Unsafe:
            self._counter_example = None

        self.max_depth = self._max_depth
        self.branches_explored = self._branches_explored
        self.status = self._status
        self.counter_example = self._counter_example

        return self.status

    def one_shot_attempt(self, objective: Objective) -> Status:

        """
        Makes a one-shot attempt at solving the verification problem without branching.

        Args:
            objective:
                The verification objective.
        Returns:
            The status object.
        """

        logger.debug(f"Starting one-shot attempt")

        objective.model.set_device(use_gpu=self._master_worker_should_use_gpu)
        solver = Verifier(objective.model, objective, use_progress_bars=False)
        status = Status.Undecided

        self._max_depth = 0
        self._branches_explored = 1

        if CONFIG.USE_PRE_PROCESSING_ATTACK:
            cex = solver.pre_process_attack()
            if cex is not None:
                status = Status.Unsafe

        if status == Status.Undecided:
            self._all_children_done.wait()
            status = solver.verify(Branch(0, None, []), finished_flag=None, needs_branches=None,
                                   put_queue=self._put_branch, queue_depth=0, no_split=True)

        if status == Status.Safe or status == Status.Unsafe:

            self._status = status

            if status == Status.Unsafe:
                self._counter_example = solver.counter_example
            else:
                self._counter_example = None

            self.max_depth = self._max_depth
            self.branches_explored = self._branches_explored
            self.status = self._status
            self.counter_example = self._counter_example

        objective.model.set_device(use_gpu=False)

        logger.debug(f"One-shot attempt finished with status: {status}")

        return status

    def _set_activate_master_worker_flags(self):

        """
        Sets the necessary flags to restart all master worker.
        """

        logger.debug(f"Main process starting master workers")

        with self._work_lock:
            self._active_workers.value += 1

        self._activate_master_worker.set()

    def _set_activate_slave_workers_flags(self):

        """
        Sets the necessary flags to restart all workers.
        """

        logger.debug(f"Main process starting all workers")

        with self._work_lock:
            self._active_workers.value += self._max_procs - 1

        self._activate_slave_workers.set()

    def _create_workers(self, num: int = None):

        """
        Creates num workers.

        The workers are stored in self._workers. It is assumed that no previous
        workers exist.

        Args:
            num:
                The number of workers that should be created. If arg is None,
                self._max_procs - self._num_workers workers are created.
        """

        num = num if num is not None else (self._max_procs - self._num_workers)

        # We don't store directly in self._workers as this would require pickling and
        # copying to all children which interferes with spawn().
        workers = self._workers if self._workers is not None else []
        self._workers = None

        for i in range(num):
            if CONFIG.PROFILE_WORKER:
                worker = self._mp_context.Process(target=self._start_worker_profile, args=())
            else:
                worker = self._mp_context.Process(target=self._run_worker, args=())
            worker.start()

            workers.append(worker)
            logger.debug(f"Added worker {i + 1} of {num}")

        self._active_workers.value += num
        self._workers = workers

    def _suspend_workers(self):

        """
        Suspends children by putting 'None' on the branch queue.

        If not all children suspend within CONFIG.MAX_CHILDREN_SUSPEND_TIME, a
        ChildProcessError is raised.
        """

        logger.debug(f"Putting poison pills")
        with self._work_lock:  # Acquire lock to make sure that workers don't queue new branches after None
            self._finished_flag.set()
            self._activate_slave_workers.clear()
            self._activate_master_worker.clear()

        for i in range(self._active_workers.value):
            with self._active_tasks_lock:
                self._active_tasks.value += 1
            logger.debug(f"Added poison pill {i + 1} of {self._max_procs}")
            self._branch_queue.put(None)  # "Poison pill for killing child processes"

        # Wait for children to empty branch and suspend
        success = self._all_children_done.wait(CONFIG.MAX_CHILDREN_SUSPEND_TIME)

        if not success:
            logger.error("At least one worker did not suspend as expected")
            self._log_mp_values()
            raise ChildProcessError("At least one worker did not suspend as expected")

    def _wait_for_workers(self, timeout: float):

        """
        Wait until children are done.

        The process will go to sleep until children are done or self.timeout is
        reached.

        Args:
            timeout:
                The timeout parameter
        Returns:
            True if timed-out before all workers finished, else False.
        """

        logger.debug("Main process waiting for workers")
        did_timeout = not self._finished_flag.wait(timeout=timeout)
        logger.debug(f"Main process finished waiting, timeout={timeout}")

        return did_timeout

    # noinspection PyArgumentList
    def _join_workers(self):

        """
        Joins all workers.
        """

        self._destroy_flag.set()
        self._activate_master_worker.set()
        self._activate_slave_workers.set()
        self._all_children_done.wait(CONFIG.MAX_CHILDREN_SUSPEND_TIME)

        logger.debug(f"Main process started joining workers")
        for worker in self._workers:
            worker.join()
            if worker.exitcode is None:
                logger.warning(f"Main process could not join with worker, terminating instead")
                worker.terminate()
            self._active_workers.value -= 1

        self._workers = None
        self._activate_slave_workers.clear()
        self._activate_master_worker.clear()
        self._destroy_flag.clear()

        logger.debug(f"Main process finished joining workers")

    def _run_worker(self):

        """
        Runs the worker.

        The worker will pick branches from the queue and verify them as long as
        self._finished_flag is not set. If it is set, the workers will empty the
        queue, without processing the items and on getting a 'None' they will
        suspend until self._activate_slave_workers is set. When self._activate_slave_workers
        is set, the process repeats.
        """

        solver = None

        while True:
            logger.debug(f"Worker {os.getpid()} requesting branch from queue")
            branch = self._branch_queue.get()
            logger.debug(f"Worker {os.getpid()} retrieved branch <addr: {hex(id(branch))}> from queue")

            if branch is None:  # Handle poison pill

                del solver
                self._suspend_worker()
                solver = self._get_solver()
                if solver is None:
                    return
                else:
                    continue

            if self._finished_flag.is_set():  # Empty multiprocessing queue

                logger.debug(f"Worker {os.getpid()} discarded branch <addr: {hex(id(branch))}>")
                with self._active_tasks_lock:
                    self._active_tasks.value -= 1
                continue

            logger.debug(f"Worker {os.getpid()} started iteration, active tasks: {self._active_tasks.value}")
            self._verify_branch(branch, solver)
            self._finished_subtree(solver.max_depth, solver.branches_explored, solver.status, solver.counter_example)
            logger.debug(f"Worker {os.getpid()} finished iteration, Status: {solver.status}, "
                         f"active tasks: {self._active_tasks.value}")

    def _verify_branch(self, branch: Branch, solver: Verifier):

        """
        Verifies the given branch

        Args:
            branch:
                The branch to be solved
            solver:
                The Verifier object
        """

        solver.verify(branch, finished_flag=self._finished_flag, needs_branches=self._needs_branches,
                      put_queue=self._put_branch, queue_depth=self._queue_depth, no_split=False)

        logger.debug(f"Worker {os.getpid()} finished solving branch <addr: {hex(id(branch))}>")

    def _suspend_worker(self):

        """
        Suspends the worker until signaled by main process.
        """

        with self._active_tasks_lock:
            self._active_tasks.value -= 1

        with self._work_lock:
            self._active_workers.value -= 1
            if self._active_workers.value == 0:
                self._all_children_done.set()

        logger.debug(f"Worker {os.getpid()} suspended, active procs: {self._active_workers.value}/{self._max_procs}")

        if os.getpid() == self._master_worker_pid.value:
            self._activate_master_worker.wait()
        else:
            self._activate_slave_workers.wait()

        logger.debug(f"Worker {os.getpid()} reactivated")

    def _get_solver(self):

        """
        Gets the solver, unless self._destroy_flag is set, in which case None is
        returned.
        """

        if self._destroy_flag.is_set():
            logger.debug(f"Worker {os.getpid()} exited")
            return None
        else:
            model, objective = self._get_solver_args()

            if os.getpid() == self._master_worker_pid.value:
                model.set_device(self._master_worker_should_use_gpu)

            return Verifier(model, objective, use_progress_bars=CONFIG.USE_PROGRESS_BARS)

    def _get_solver_args(self) -> tuple:

        """
        Helper function, extracting necessary solver args from multiprocessing queue.

        Returns:
            A tuple with the args
        """

        objective = pickle.loads(self.objective)
        return objective.model, objective

    def _finished_subtree(self, max_depth: int, branches_explored: int,
                          status: Status, counter_example: np.array = None):

        """
        Called from workers when they finish their current subtree.

        This function updates the relevant statistics, such as max_depth and branches
        explored. If a worker finds the system to be unsafe or encounters underflow,
        the found solution flag is set.

        Args:
            max_depth:
                The workers maximum branch depth
            branches_explored:
                The number of branches the worker explored
            status:
                The Status Enum with the final status of the subtree
            counter_example:
                The counter example, if found.
        """

        with self._work_lock:

            self._max_depth = max_depth if max_depth > self._max_depth else self._max_depth
            self._branches_explored += branches_explored

            if not self._finished_flag.is_set() and status == Status.Unsafe:
                self._status = status
                self._counter_example = counter_example
                self._set_solution_found_flags()

            elif not self._finished_flag.is_set() and status == Status.Underflow:
                self._status = status
                self._set_solution_found_flags()

        with self._active_tasks_lock:
            self._active_tasks.value -= 1
            if self._active_tasks.value == 0:
                self._set_solution_found_flags()

    def _set_solution_found_flags(self):

        """
        Sets the necessary flags to tell other procs that a solution has been found
        """

        self._finished_flag.set()
        self._activate_slave_workers.clear()
        self._activate_master_worker.clear()

    def _needs_branches(self) -> bool:

        """
        Returns true if the queue needs more branches.
        """

        return self._active_tasks.value < (2 * self._max_procs)

    def _put_branch(self, branch: Branch):

        """
        This method is called by workers to put a branch into the queue.

        Args:
            branch:
                The new branch.
        """

        logger.debug(f"Worker {os.getpid()} started putting branch <addr: {hex(id(branch))}> on queue")

        with self._work_lock:
            if not self._finished_flag.is_set():
                with self._active_tasks_lock:
                    self._active_tasks.value += 1
                if not self._finished_flag.is_set():
                    self._branch_queue.put(branch)

        logger.debug(f"Worker {os.getpid()} finished putting branch <addr: {hex(id(branch))}> on queue")

    def _handle_memory(self):

        """
        This function kills all workers if memory usage is too large.

        VeriNet has a small memory leak, leading to increased memory usage over time.
        This can become significant if when verifying a large amount of problems for
        large networks. This function kills all child processes if memory usage is
        has increased significantly since the process started.

        A lot of work has gone into locating the memory leak; however, the tracemalloc
        module indicates that there is no significant leak. Since tracemalloc doesn't
        locate the leak, it most likely happens in the c-backend of one or more
        libraries (numpy, numba, multiprocessing, Xpress...). So, this was the
        second-best solution and creates only minimal overhead.
        """

        logger.debug(f"Memory usage: {psutil.virtual_memory()[2]}")

        if self._initial_mem_usage is None:
            self._initial_mem_usage = psutil.virtual_memory()[2]

        elif (psutil.virtual_memory()[2] - self._initial_mem_usage) > CONFIG.MAX_ACCEPTED_MEMORY_INCREASE:
            logger.debug(f"Max accepted memory increase exceeded, performing cleanup.")
            self.cleanup()
            self._branch_queue = self._mp_context.Manager().Queue()
            self._initial_mem_usage = psutil.virtual_memory()[2]
            self._init_workers()

    def _reset_params(self):

        """
        Resets all statistics stored from the last verification run.
        """

        self._max_depth = 0
        self._branches_explored = 0
        self._status = Status.Undecided
        self._counter_example = None

        self._finished_flag.clear()
        self._destroy_flag.clear()

    def _log_mp_values(self):

        """
        Logs some multiprocessing values.
        """

        logger.debug(f"Active procs: {self._active_workers.value}")
        logger.debug(f"Active tasks: {self._active_tasks.value}")
        logger.debug(f"Finished flag: {self._finished_flag.is_set()}")
        logger.debug(f"Destroy flag: {self._destroy_flag.is_set()}")
        logger.debug(f"All workers done flag: {self._all_children_done.is_set()}")
        logger.debug(f"Activate master worker flag: {self._activate_master_worker.is_set()}")
        logger.debug(f"Activate all workers flag: {self._activate_slave_workers.is_set()}")

    def _start_worker_profile(self):

        """
        Used for profiling worker processes.
        """

        path = "../../profiling/"
        if not os.path.isdir(path):
            os.mkdir(path)

        self._profile(self._run_worker, filepath=os.path.join(path, f'profile_worker_{os.getpid()}.txt'), args=())

    # noinspection PyMethodMayBeStatic
    def _profile(self, func: callable, filepath: str, args: tuple = ()):

        """
        A wrapper for profiling methods.

        Methods started with this wrapper are run with cProfiling and the
        profile is stored in ../../profiling.

        Args:
            func:
                The function that should be profiled.
            filepath:
                The path where the profile file is stored.
            args:
                The args used for calling the given function.
        """

        from cProfile import Profile
        import pstats

        pr = Profile()
        pr.enable()
        func(*args)
        pr.disable()

        with open(filepath, 'w+') as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs().sort_stats('cumtime').print_stats()

    # noinspection PyArgumentList
    def cleanup(self):

        """
        Kills all child-processes
        """

        if os.getpid() == self._main_pid:

            if self._workers is not None:
                self._join_workers()
                self._cleanup_multiprocessing_resource_trackers()

            if self._branch_queue is not None:
                self._branch_queue = None

        self._workers = None

    @staticmethod
    def _cleanup_multiprocessing_resource_trackers():

        """
        Force terminates resource tracker processes.

        Forkserver and Spawn methods for creating child processes also create
        processes to track shared resources such as semaphores.

        These processes don't always terminate correctly leading to large memory
        leaks. (Seems to be an issue related to Python 3.8, see
        https://bugs.python.org/issue38842).

        This method is a crude fix forcefully terminating the processes; however
        note that this can lead to a leak in semaphore locks that may not be
        freed until reboot.
        """

        if CONFIG.CHILDPROC_TYPE == "spawn":
            os.system('kill -9 $(pgrep -f "multiprocessing.spawn")')
        elif CONFIG.CHILDPROC_TYPE == "forkserver":
            os.system('kill -9 $(pgrep -f "multiprocessing.forkserver")')
