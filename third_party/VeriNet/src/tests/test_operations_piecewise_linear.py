
"""
Unittests for the piecewise linear operations

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

import torch
import torch.nn as nn

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.operations.piecewise_linear import Relu


# noinspection PyCallingNonCallable
class TestOperationsPiecewiseLinear(unittest.TestCase):

    def setUp(self):

        self.relu = Relu()

    def test_relu_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.relu.is_monotonically_increasing)
        self.assertFalse(self.relu.is_linear)
        self.assertTrue(nn.modules.activation.ReLU in self.relu.abstracted_torch_funcs())
        self.assertTrue(nn.ReLU in self.relu.abstracted_torch_funcs())

    # noinspection PyArgumentList,PyTypeChecker
    def test_relu_forward(self):

        """
        Test the forward method.
        """

        x = torch.FloatTensor([2.5, -2.5])
        res = self.relu.forward(x)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0], 2.5)
        self.assertAlmostEqual(res[1], 0)

    # noinspection PyArgumentList,PyTypeChecker
    def test_relu_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        relax = self.relu.linear_relaxation(torch.FloatTensor([-2.5]), torch.FloatTensor([-1]))[1]

        self.assertTrue(isinstance(relax, torch.FloatTensor))
        self.assertAlmostEqual(relax[0, 0], 0)
        self.assertAlmostEqual(relax[0, 1], 0)

        relax = self.relu.linear_relaxation(torch.FloatTensor([1]), torch.FloatTensor([2.5]))[1]

        self.assertTrue(isinstance(relax, torch.FloatTensor))
        self.assertAlmostEqual(relax[0, 0], 1)
        self.assertAlmostEqual(relax[0, 1], 0)

        relax = self.relu.linear_relaxation(torch.FloatTensor([-1]), torch.FloatTensor([1]))[1]

        self.assertTrue(isinstance(relax, torch.FloatTensor))
        self.assertAlmostEqual(relax[0, 0], 0.5)
        self.assertAlmostEqual(relax[0, 1], 0.5)

    def test_relu_split_point(self):

        """
        Test the split_point() method.
        """

        split_point = self.relu.split_point(-2.5, 2)

        self.assertAlmostEqual(split_point, 0)

    def test_backprop_through_relaxation_relu(self):

        """
        Tests the _backprop_through_relaxation_relu method.
        """

        bounds_symbolic_post = torch.FloatTensor([[1, 2, 1],
                                                  [-1, -2, 0]])

        bounds_concrete_pre = torch.FloatTensor([[0, 10], [-10, 10]])
        relaxations = torch.FloatTensor([[[0.2, 0], [0.1, 0]],
                                         [[2, 2], [0.5, 1]]])

        res, _, _, _ = self.relu.backprop_through_relaxation(bounds_symbolic_post=bounds_symbolic_post,
                                                             bounds_concrete_pre=bounds_concrete_pre,
                                                             relaxations=relaxations,
                                                             lower=True)
        gt = torch.FloatTensor([[1, 0.2, 1], [-1, -1, -2]])

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertEqual(float(res[0][i, j]), float(gt[i, j]))
