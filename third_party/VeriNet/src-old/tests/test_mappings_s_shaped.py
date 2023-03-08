
"""
Unit-tests for the S-shaped mappings

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import numpy as np

from src.algorithm.mappings.s_shaped import Sigmoid, Tanh


class TestMappingSShaped(unittest.TestCase):

    def setUp(self):

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    @staticmethod
    def s(x):

        """
        Sigmoid function used for calculating ground trouth values.
        """

        return 1/(1 + np.exp(-x))

    def test_sigmoid_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.sigmoid.is_1d_to_1d)
        self.assertFalse(self.sigmoid.is_linear)

    def test_sigmoid_propagate_float(self):

        """
        Test the propagate() method with floats.
        """

        x0, x1 = 2.5, -2.5

        self.assertAlmostEqual(self.sigmoid.propagate(x0), self.s(x0))
        self.assertAlmostEqual(self.sigmoid.propagate(x1), self.s(x1))

    def test_sigmoid_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([2.5, -2.5])
        res = self.sigmoid.propagate(x)

        self.assertAlmostEqual(res[0], self.s(x[0]))
        self.assertAlmostEqual(res[1], self.s(x[1]))

    def test_sigmoid_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        # Test intercepting line

        bounds = np.array([[-2.5, -1]])
        relax = self.sigmoid.linear_relaxation(bounds[:, 0], bounds[:, 1], upper=True)
        gt_a = (self.s(bounds[0, 1]) - self.s(bounds[0, 0]))/(bounds[0, 1] - bounds[0, 0])
        gt_b = self.s(bounds[0, 1]) - (gt_a * bounds[0, 1])
        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

        # Test optimal tangent:

        bounds = np.array([[0, 2.5]])
        relax = self.sigmoid.linear_relaxation(bounds[:, 0], bounds[:, 1], upper=True)
        tangent_point = ((bounds[:, 1]**2 - bounds[:, 0]**2)/(2*(bounds[:, 1] - bounds[:, 0])))[0]
        gt_a = self.s(tangent_point) * (1-self.s(tangent_point))
        gt_b = self.s(tangent_point) - (gt_a * tangent_point)

        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

    def test_sigmoid_split_point(self):

        """
        Test the split_point() method.
        """

        split_point = self.sigmoid.split_point(-2.5, 2)
        mid = (self.s(2) + self.s(-2.5)) / 2
        self.assertAlmostEqual(split_point, -np.log((1 / mid) - 1))

    def test_tanh_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.tanh.is_1d_to_1d)
        self.assertFalse(self.tanh.is_linear)

    def test_tanh_propagate_float(self):

        """
        Test the propagate() method with floats.
        """

        x0, x1 = 2.5, -2.5

        self.assertAlmostEqual(self.tanh.propagate(x0), np.tanh(x0))
        self.assertAlmostEqual(self.tanh.propagate(x1), np.tanh(x1))

    def test_tanh_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([2.5, -2.5])
        res = self.tanh.propagate(x)

        self.assertAlmostEqual(res[0], np.tanh(x[0]))
        self.assertAlmostEqual(res[1], np.tanh(x[1]))

    def test_tanh_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        # Test intercepting line

        bounds = np.array([[-2.5, -1]])
        relax = self.tanh.linear_relaxation(bounds[:, 0], bounds[:, 1], upper=True)
        gt_a = (np.tanh(bounds[0, 1]) - np.tanh(bounds[0, 0]))/(bounds[0, 1] - bounds[0, 0])
        gt_b = np.tanh(bounds[0, 1]) - (gt_a * bounds[0, 1])
        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

        # Test optimal tangent:

        bounds = np.array([[0, 2.5]])
        relax = self.tanh.linear_relaxation(bounds[:, 0], bounds[:, 1], upper=True)
        tangent_point = ((bounds[:, 1]**2 - bounds[:, 0]**2)/(2*(bounds[:, 1] - bounds[:, 0])))[0]
        gt_a = 1 - np.tanh(tangent_point)**2
        gt_b = np.tanh(tangent_point) - (gt_a * tangent_point)

        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

    def test_tanh_split_point(self):
        """
        Test the split_point() method.
        """

        split_point = self.tanh.split_point(-2.5, 2)

        mid = (np.tanh(2) + np.tanh(-2.5)) / 2
        self.assertAlmostEqual(split_point, 0.5 * np.log((1 + mid) / (1 - mid)))
