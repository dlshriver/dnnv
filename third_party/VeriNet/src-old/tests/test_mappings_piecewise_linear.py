
"""
Unit-tests for the piecewise linear mappings

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import numpy as np

from src.algorithm.mappings.piecewise_linear import Relu


class TestMappingPiecewiseLinear(unittest.TestCase):

    def setUp(self):

        self.relu = Relu()

    def test_relu_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.relu.is_1d_to_1d)
        self.assertFalse(self.relu.is_linear)

    def test_relu_propagate_float(self):

        """
        Test the propagate() method with floats.
        """

        x0, x1 = 2.5, -2.5

        self.assertAlmostEqual(self.relu.propagate(x0), 2.5)
        self.assertAlmostEqual(self.relu.propagate(x1), 0)

    def test_relu_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([2.5, -2.5])
        res = self.relu.propagate(x)

        self.assertAlmostEqual(res[0], 2.5)
        self.assertAlmostEqual(res[1], 0)

    def test_relu_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        relax = self.relu.linear_relaxation(np.array([-2.5]), np.array([-1]), upper=True)

        self.assertAlmostEqual(relax[0, 0], 0)
        self.assertAlmostEqual(relax[0, 1], 0)

        relax = self.relu.linear_relaxation(np.array([1]), np.array([2.5]), upper=True)

        self.assertAlmostEqual(relax[0, 0], 1)
        self.assertAlmostEqual(relax[0, 1], 0)

        relax = self.relu.linear_relaxation(np.array([-1]), np.array([1]), upper=True)

        self.assertAlmostEqual(relax[0, 0], 0.5)
        self.assertAlmostEqual(relax[0, 1], 0.5)

    def test_relu_split_point(self):

        """
        Test the split_point() method.
        """

        split_point = self.relu.split_point(-2.5, 2)

        self.assertAlmostEqual(split_point, 0)
