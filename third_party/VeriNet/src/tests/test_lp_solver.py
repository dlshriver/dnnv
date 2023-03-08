
"""
Unittests for the LP-Solver class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import warnings
import logging

import numpy as np

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.verification.lp_solver import LPSolver


class TestLPSolver(unittest.TestCase):

    def setUp(self):

        """
        Running the unittest triggers two warnings:

        - sys:1: DeprecationWarning: PyArray_GetNumericOps is deprecated.
        - sys:1: DeprecationWarning: PyArray_SetNumericOps is deprecated. Use
          PyUFunc_ReplaceLoopBySignature to replace ufunc inner loop functions
          instead.

        The warnings seem to be triggered when importing LPSolver; however, this
        behaviour is not observed during normal usage. The methods in question
        seem to be numpy-methods and might be related to the Xpress-numpy
        interface. Since the warnings are only triggered during unittests, we
        filter them out.
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        self._input_size = 2
        self._lp_solver = LPSolver(input_size=self._input_size)

    def test_init(self):

        """
        Tests that the variables are initialised correctly.
        """

        self.assertEqual(self._lp_solver._input_size, self._input_size)
        self.assertEqual(len(self._lp_solver.variables), self._input_size)

    def test_add_bias_variables(self):

        """
        Tests bias variables are added.
        """

        self._lp_solver.add_bias_variables(10)

        self.assertEqual(len(self._lp_solver._bias_variables), 10)
        self.assertEqual(len(self._lp_solver.variables), 12)

        self._lp_solver.add_bias_variables(5)

        self.assertEqual(len(self._lp_solver._bias_variables), 15)
        self.assertEqual(len(self._lp_solver.variables), 17)

    def test_remove_variables(self):

        """
        Test that _remove_variables() removes all variables.
        """

        self._lp_solver.add_bias_variables(10)
        self._lp_solver._remove_variables()
        self.assertIsNone(self._lp_solver._input_variables)
        self.assertIsNone(self._lp_solver._bias_variables)
        self.assertEqual(len(self._lp_solver._solver.getVariable()), 0)

    def test_set_input_bounds(self):

        """
        tests that the input bounds are set correctly.
        """

        bounds = np.array([[0, 1], [1, 2]])

        self._lp_solver.set_input_bounds(bounds)

        lb = []
        ub = []
        self._lp_solver._solver.getlb(lb)
        self._lp_solver._solver.getub(ub)

        for i in range(self._input_size):
            self.assertEqual(lb[i], bounds[i, 0])
            self.assertEqual(ub[i], bounds[i, 1])

    def test_add_constraints(self):

        """
        Tests that the constraints are added correctly.
        """

        self._lp_solver.add_constraints(coeffs=np.array([[1, 2], [2, 3]]), constants=np.array([1, 2]),
                                        constr_types=['L', 'G'])

        self.assertEqual(2, len(self._lp_solver._solver.getConstraint()))

    def test_remove_constraints(self):

        """
        Tests that constraints are removed correctly.
        """

        constraints = self._lp_solver.add_constraints(coeffs=np.array([[1, 2], [2, 3]]), constants=np.array([1, 2]),
                                                      constr_types=['L', 'G'])

        self._lp_solver.remove_constraints([constraints[0]])
        self.assertEqual(constraints[1], self._lp_solver._solver.getConstraint()[0])

    def test_solve(self):

        """
        Checks that the solve-method returns feasible/ infeasible correctly for two
        simple test-cases.
        """

        bounds = np.array([[0, 1], [1.5, 2]])
        self._lp_solver.set_input_bounds(bounds)

        self.assertTrue(self._lp_solver.solve())
        self._lp_solver.add_constraints(coeffs=np.array([[1, -1]]), constants=np.array([0]),
                                        constr_types=['G'])
        self.assertFalse(self._lp_solver.solve())

    def test_get_all_constraints(self):

        """
        Tests the get_all_constraints method.
        """

        self.assertEqual(len(self._lp_solver.get_all_constraints()), 0)

        self._lp_solver.add_constraints(coeffs=np.array([[1, 2], [2, 3]]), constants=np.array([1, 2]),
                                        constr_types=['L', 'G'])

        self.assertEqual(len(self._lp_solver.get_all_constraints()), 2)
