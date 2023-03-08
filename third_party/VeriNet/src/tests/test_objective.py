
"""
Unittests for the Objective class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

import numpy as np
import torch

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.tests.simple_nn import SimpleNN
from verinet.verification.objective import Objective
from verinet.verification.lp_solver import LPSolver
from verinet.sip_torch.rsip import RSIP
from verinet.tests.simple_nn import SimpleNN2Outputs
from verinet.verification.verifier_util import Branch, Status


# noinspection DuplicatedCode
class TestLPSolver(unittest.TestCase):

    def setUp(self):

        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = False
        input_bounds = np.array([[[1, 2], [3, 4]]])
        self._objective = Objective(input_bounds, output_size=2, model=SimpleNN("ReLU"))

    def test_invalid_bounds(self):

        input_bounds = np.array([[[1, 2], [4, 3]]])
        try:
            self._objective = Objective(input_bounds, output_size=2, model=SimpleNN("ReLU"))
            self.assertTrue(False)  # Should throw error.
        except ValueError:
            pass

    def test_init(self):

        """
        Tests that some variables in init are initialised correctly.
        """

        self.assertEqual(len(self._objective.input_shape), 3)
        self.assertEqual(self._objective.input_shape[0], 1)
        self.assertEqual(self._objective.input_shape[1], 1)
        self.assertEqual(self._objective.input_shape[2], 2)
        self.assertEqual(self._objective.input_bounds_flat.shape[0], 2)
        self.assertEqual(self._objective.input_bounds_flat.shape[1], 2)
        self.assertEqual(self._objective.output_size, 2)
        self.assertEqual(len(self._objective.output_vars), 2)

    def test_add_constraints(self):

        """
        Tests that two constraints are added correctly with the add_constraints() method
        """

        y = self._objective.output_vars

        c0 = y[0] + y[1] <= 0
        c1 = y[0] - y[1] >= 1

        self._objective.add_constraints([c0, c1])

        self.assertTrue(c0, self._objective._constraints[0])
        self.assertTrue(c1, self._objective._constraints[1])

    def test_remove_constraints(self):

        """
        Tests that two constraints are added correctly with the add_constraints() method
        """

        y = self._objective.output_vars

        c0 = y[0] + y[1] <= 0
        c1 = y[0] - y[1] >= 1

        self._objective.add_constraints([c0, c1])
        self._objective.remove_constraints(c0)

        self.assertTrue(c1, self._objective._constraints[0])
        self.assertTrue(1, len(self._objective._constraints))

    # noinspection PyCallingNonCallable
    def test_grad_descent_losses(self):

        """
        Tests that two constraints are added correctly with the add_constraints() method
        """

        y = self._objective.output_vars

        c0 = y[0] - y[1] >= 1
        self._objective.add_constraints([c0])
        self._objective._current_constraint_idx = 0

        out = torch.Tensor([[2, 3]])

        self.assertTrue(-1, self._objective.grad_descent_loss(out))

    # noinspection PyCallingNonCallable
    def test_is_counter_example(self):

        """
        Tests the is_counter_example() method with two simple cases
        """

        y = self._objective.output_vars

        c0 = y[0] - y[1] >= 1
        self._objective.add_constraints([c0])
        self._objective._current_constraint_idx = 0

        out1 = torch.FloatTensor([[2, 3]])
        out2 = torch.FloatTensor([[4, 2]])

        self.assertTrue(self._objective.is_counter_example(out1))
        self.assertFalse(self._objective.is_counter_example(out2))

    # noinspection PyCallingNonCallable
    def test_find_potential_cex_inner(self):

        """
        Tests the _find_potential_cex method.
        """

        LPSolver.__del__ = lambda x: None

        model = SimpleNN2Outputs("ReLU")
        solver = LPSolver(2)
        sip = RSIP(model, input_shape=torch.LongTensor((2,)))

        y = self._objective.output_vars
        c0 = y[0] - y[1] >= 0
        self._objective.add_constraints([c0])

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        solver.set_input_bounds(input_bounds.numpy())
        sip.calc_bounds(input_bounds)

        self._objective.cleanup(solver)

        solver.add_constraints(coeffs=np.array([[1, 0]]), constants=np.array([0]), constr_types=['G'])
        res, _ = self._objective._find_potential_cex(solver, sip, use_optimised_relaxation_constraints=False)

        self.assertGreaterEqual(res[0], 0)
        self.assertGreaterEqual(res[1], -1)

        solver.remove_all_constraints()
        solver.add_constraints(coeffs=np.array([[1, 0], [0, 1]]), constants=np.array([-1, -1]), constr_types=['G', 'G'])

        res, _ = self._objective._find_potential_cex(solver, sip, use_optimised_relaxation_constraints=False)

        self.assertIsNone(res)

    # noinspection PyCallingNonCallable
    def test_find_potential_cex_simple(self):

        """
        Tests the _find_potential_cex_simple method.
        """

        model = SimpleNN2Outputs("ReLU")
        sip = RSIP(model, input_shape=torch.LongTensor((2,)))
        solver = LPSolver(2)

        y = self._objective.output_vars
        c0 = y[0] - y[1] >= 0
        self._objective.add_constraints([c0])

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        sip.calc_bounds(input_bounds)

        self._objective.cleanup(solver)

        res, _ = self._objective._find_potential_cex_simple(sip, use_optimised_relaxation_constraints=False)

        self.assertGreaterEqual(res[0], -1)
        self.assertGreaterEqual(res[1], -1)

    # noinspection PyCallingNonCallable,PyTypeChecker
    def test_find_potential_cex_simple_2_constraints(self):

        """
        Tests the _find_potential_cex_simple method with two constraints .
        """

        model = SimpleNN2Outputs("ReLU")
        sip = RSIP(model, input_shape=torch.LongTensor((2,)))
        solver = LPSolver(2)

        y = self._objective.output_vars
        c0 = y[0] - y[1] >= 0
        c1 = y[0] >= 2.5
        self._objective.add_constraints([c0, c1])

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        sip.calc_bounds(input_bounds)

        self._objective.cleanup(solver)

        res, _ = self._objective._find_potential_cex_simple(sip, use_optimised_relaxation_constraints=False)

        self.assertGreaterEqual(res[0], -1)
        self.assertGreaterEqual(res[1], -1)

        self._objective.finished_constraint(None, Status.Safe)
        res, _ = self._objective._find_potential_cex_simple(sip, use_optimised_relaxation_constraints=False)

        self.assertAlmostEqual(float(res[0]), -1)
        self.assertAlmostEqual(float(res[1]), -1)

    # noinspection PyCallingNonCallable
    def test_find_potential_cex_outer(self):

        """
        Tests the find_potential_cex method.
        """

        LPSolver.__del__ = lambda x: None

        model = SimpleNN2Outputs("ReLU")
        branch = Branch(0, None, [{"node": 0, "neuron": 1, "split_x": 0, "upper": True}])
        solver = LPSolver(2)
        sip = RSIP(model, input_shape=torch.LongTensor((2,)))

        y = self._objective.output_vars
        c0 = y[0] - y[1] >= 0
        self._objective.add_constraints([c0])

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        solver.set_input_bounds(input_bounds.numpy())
        sip.calc_bounds(input_bounds)

        self._objective.cleanup(solver)
        branch._split_list = [{"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        _, res, _ = self._objective.find_potential_cex(branch, solver, sip)

        self.assertGreaterEqual(res[0], -1)
        self.assertGreaterEqual(res[1], -1)

        solver.remove_all_constraints()
        solver.add_constraints(coeffs=np.array([[1, 0], [0, 1]]), constants=np.array([-1, -1]), constr_types=['G', 'G'])

        _, res, _ = self._objective.find_potential_cex(branch, solver, sip)

        self.assertIsNone(res)

    # noinspection PyCallingNonCallable
    def test_find_potential_cex_outer_2_constraints(self):

        """
        Tests the find_potential_cex method with two constraints.
        """

        LPSolver.__del__ = lambda x: None

        model = SimpleNN2Outputs("ReLU")
        branch = Branch(0, None, [{"node": 1, "neuron": 1, "split_x": 0, "upper": True}])
        solver = LPSolver(2)
        sip = RSIP(model, input_shape=torch.LongTensor((2,)))

        y = self._objective.output_vars
        c0 = y[0] - y[1] >= 0
        c1 = y[0] >= 2.5
        self._objective.add_constraints([c0, c1])

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        solver.set_input_bounds(input_bounds.numpy())
        sip.calc_bounds(input_bounds)

        self._objective.cleanup(solver)

        solver.add_constraints(coeffs=np.array([[1, 0], [0, 1]]),
                               constants=np.array([-0.5, -0.5]), constr_types=['G', 'G'])

        _, res, _ = self._objective.find_potential_cex(branch, solver, sip)

        self.assertIsNone(res)

        self._objective.finished_constraint(solver, Status.Safe)
        _, res, _ = self._objective.find_potential_cex(branch, solver, sip)

        self.assertAlmostEqual(float(res[0]), 0.5)
        self.assertAlmostEqual(float(res[1]), 0.5)

    # noinspection PyCallingNonCallable
    def test_maximise_eq(self):

        """
        Tests the _maximise_eq method.
        """

        equation = torch.FloatTensor([1, -2, 3])
        bounds = torch.FloatTensor([[-1, 1], [-2, 0], [0, 3]])

        eval_point, max_value = self._objective.maximise_eq(equation, bounds)
        gt_eval_point = [1, -2, 3]
        gt_max_value = 14

        for i in range(len(gt_eval_point)):
            self.assertEqual(float(eval_point[i]), gt_eval_point[i])

        self.assertAlmostEqual(float(max_value), gt_max_value)

    # noinspection PyCallingNonCallable
    def test_finished_constraint(self):

        """
        Tests the finished_constraint method.
        """

        LPSolver.__del__ = lambda x: None
        solver = LPSolver(2)
        self._objective.cleanup(solver)

        lp_solver_constraint1 = solver.add_constraints(coeffs=np.array([1, 0]),
                                                       constants=np.array([1]),
                                                       constr_types=['G'])[0]
        lp_solver_constraint2 = solver.add_constraints(coeffs=np.array([1, 0]),
                                                       constants=np.array([1]),
                                                       constr_types=['G'])[0]

        self._objective._active_lp_solver_constraints = [lp_solver_constraint1]
        self._objective.finished_constraint(solver, Status.Safe)
        self._objective.finished_constraint(solver, Status.Unsafe)

        self.assertTrue(solver.get_all_constraints()[0] == lp_solver_constraint2)
        self.assertEqual(self._objective.safe_constraints[0],  0)
