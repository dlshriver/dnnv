
"""
Unittests for the Branch class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import torch
import numpy as np
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.util.config import CONFIG
from verinet.verification.verifier_util import Branch
from verinet.verification.lp_solver import LPSolver
from verinet.sip_torch.rsip import RSIP
from verinet.tests.simple_nn import SimpleNN


# noinspection PyCallingNonCallable
class TestBranch(unittest.TestCase):

    def setUp(self):

        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = False

        self.branch = Branch(0, None, [])

        self.model = SimpleNN("ReLU")
        self.sip = RSIP(self.model, input_shape=torch.LongTensor((2,)))
        self.solver = LPSolver(2)

        input_bounds = torch.FloatTensor([[-1, 1], [-1, 1]])
        self.sip.calc_bounds(input_bounds)

    def test_init(self):

        """
        Tests that some variables in init are initialised correctly.
        """

        self.assertEqual(self.branch.depth, 0)
        self.assertIsNone(self.branch.forced_bounds_pre)
        self.assertIsNone(self.branch.lp_solver_constraints)
        self.assertEqual(len(self.branch.split_list), 0)
        self.assertEqual(len(self.branch.safe_constraints), 0)

    def test_add_constraints_to_solver(self):

        """
        Tests the add_constraints_to_solver method.
        """

        splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                  {"node": 1, "neuron": 1, "split_x": 0, "upper": False}]

        self.branch.add_constraints_to_solver(self.sip, self.solver, splits=splits)

        self.assertEqual(len(self.solver.get_all_constraints()), 2)

        const = self.solver.add_constraints(coeffs=np.array([[1, 1]]), constants=np.array([-1]), constr_types=['G'])
        self.assertTrue(self.solver.solve())

        self.solver.remove_constraints(const)

        self.solver.add_constraints(coeffs=np.array([[1, 1]]), constants=np.array([-1.1]), constr_types=['G'])
        self.assertFalse(self.solver.solve())

    def test_add_constraints_to_solver_with_bias_split(self):

        """
        Tests the add_constraints_to_solver method with bias split enabled.
        """

        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = True

        self.sip.calc_bounds(torch.FloatTensor([[-1, 1], [-1, 1]]))
        self.solver.add_bias_variables(2)

        splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                  {"node": 1, "neuron": 1, "split_x": 0, "upper": False}]

        self.branch.add_constraints_to_solver(self.sip, self.solver, splits=splits)

        self.assertEqual(len(self.solver.get_all_constraints()), 4)

    def test_add_all_constraints(self):

        """
        Tests the add_all_constraints method.
        """

        splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                  {"node": 1, "neuron": 1, "split_x": 0, "upper": False}]
        self.branch.add_all_constrains(self.sip, self.solver, splits)

        self.assertEqual(len(self.branch.lp_solver_constraints), 2)

        self.assertEqual(len(self.solver.get_all_constraints()), 2)

        const = self.solver.add_constraints(coeffs=np.array([[1, 1]]), constants=np.array([-1]), constr_types=['G'])
        self.assertTrue(self.solver.solve())

        self.solver.remove_constraints(const)

        self.solver.add_constraints(coeffs=np.array([[1, 1]]), constants=np.array([-1.1]), constr_types=['G'])
        self.assertFalse(self.solver.solve())

    def test_get_earliest_changed_node_num_descending(self):

        """
        Tests the _get_earliest_changed_node_num method on descending.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.assertEqual(self.branch._get_earliest_changed_node_num(5, old_splits), 1)

    def test_get_earliest_changed_node_num_backtrack(self):
        """
        Tests the _get_earliest_changed_node_num method on backtrack.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 2
        self.branch._split_list = new_splits

        self.assertEqual(self.branch._get_earliest_changed_node_num(5, old_splits), 1)

    def test_get_earliest_changed_node_num_backtrack_2(self):
        """
        Tests the _get_earliest_changed_node_num method on backtrack.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 2, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 3, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.assertEqual(self.branch._get_earliest_changed_node_num(5, old_splits), 1)

    def test_update_lp_solver_constraints_descending_1(self):

        """
        Tests the _update_lp_solver_constraints method on descent.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints

        self.branch._update_lp_solver_constraints(self.sip, self.solver, old_constraints, [0, 1])

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints:
            self.assertTrue(old_constr not in self.branch.lp_solver_constraints)

    def test_update_lp_solver_constraints_descending_2(self):

        """
        Tests the _update_lp_solver_constraints method on descent.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 3, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints

        self.branch._update_lp_solver_constraints(self.sip, self.solver, old_constraints, [])

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints:
            self.assertTrue(old_constr in self.branch.lp_solver_constraints)

    def test_update_lp_solver_constraints_backtrack_1(self):

        """
        Tests the _update_lp_solver_constraints method on backtrack.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 2, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints

        self.branch._update_lp_solver_constraints(self.sip, self.solver, old_constraints, [0, 1])

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints:
            self.assertFalse(old_constr in self.branch.lp_solver_constraints)

    def test_update_lp_solver_constraints_backtrack_2(self):

        """
        Tests the _update_lp_solver_constraints method on backtrack.
        """

        old_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 2, "split_x": 0, "upper": False}]

        new_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints

        self.branch._update_lp_solver_constraints(self.sip, self.solver, old_constraints, [])

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints[:2]:
            self.assertTrue(old_constr in self.branch.lp_solver_constraints)

    def test_update_constraints_descending_1(self):

        """
        Tests the update_constraints method on descent.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints
        self.branch.lp_solver_constraints = None

        self.branch.update_constraints(self.sip, self.solver, old_splits, old_constraints)

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints[:2]:
            self.assertFalse(old_constr in self.branch.lp_solver_constraints)

    def test_update_constraints_descending_2(self):

        """
        Tests the update_constraints method on descent.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 3, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints
        self.branch.lp_solver_constraints = None

        self.branch.update_constraints(self.sip, self.solver, old_splits, old_constraints)

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints[:2]:
            self.assertTrue(old_constr in self.branch.lp_solver_constraints)

    def test_update_constraints_backtrack_1(self):

        """
        Tests the update_constraints method on backtrack.
        """

        old_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 1, "neuron": 2, "split_x": 0, "upper": False}]

        new_splits = [{"node": 2, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints
        self.branch.lp_solver_constraints = None

        self.branch.update_constraints(self.sip, self.solver, old_splits, old_constraints)

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints[:2]:
            self.assertFalse(old_constr in self.branch.lp_solver_constraints)

    def test_update_constraints_backtrack_2(self):

        """
        Tests the update_constraints method on backtrack.
        """

        old_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 2, "split_x": 0, "upper": False}]

        new_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints
        self.branch.lp_solver_constraints = None

        self.branch.update_constraints(self.sip, self.solver, old_splits, old_constraints)

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)

        for old_constr in old_constraints[:2]:
            self.assertTrue(old_constr in self.branch.lp_solver_constraints)

    def test_update_constraints_backtrack_with_bias_split(self):

        """
        Tests the add_constraints_to_solver method with bias split enabled.
        """

        CONFIG.USE_BIAS_SEPARATED_CONSTRAINTS = True

        self.sip.calc_bounds(torch.FloatTensor([[-1, 1], [-1, 1]]))
        self.solver.add_bias_variables(2)

        old_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": False},
                      {"node": 2, "neuron": 2, "split_x": 0, "upper": False}]

        new_splits = [{"node": 1, "neuron": 0, "split_x": 0, "upper": True},
                      {"node": 1, "neuron": 1, "split_x": 0, "upper": True},
                      {"node": 2, "neuron": 1, "split_x": 0, "upper": True}]

        self.branch._depth = 3
        self.branch._split_list = new_splits

        self.branch.add_all_constrains(self.sip, self.solver, old_splits)
        old_constraints = self.branch.lp_solver_constraints
        self.branch.lp_solver_constraints = None

        self.branch.update_constraints(self.sip, self.solver, old_splits, old_constraints)

        self.assertEqual(len(self.branch.lp_solver_constraints), 3)
        self.assertEqual(len(self.solver.get_all_constraints()), 6)

        for old_constr in old_constraints[:2]:
            self.assertTrue(old_constr in self.branch.lp_solver_constraints)
