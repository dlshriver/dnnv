
"""
Unittests for the S-shaped operations

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import torch
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.operations.s_shaped import Sigmoid, Tanh


# noinspection PyTypeChecker,PyArgumentList,PyCallingNonCallable
class TestOperationsSShaped(unittest.TestCase):

    def setUp(self):

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    # noinspection PyTypeChecker
    @staticmethod
    def s(x):

        """
        Sigmoid function used for calculating ground truth values.
        """

        return 1/(1 + torch.exp(-x))

    def test_sigmoid_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.sigmoid.is_monotonically_increasing)
        self.assertFalse(self.sigmoid.is_linear)

    # noinspection PyTypeChecker,PyArgumentList
    def test_sigmoid_forward_float(self):

        """
        Test the forward method with floats.
        """

        x0, x1 = torch.FloatTensor([2.5]), torch.FloatTensor([-2.5])

        self.assertAlmostEqual(self.sigmoid.forward(x0), self.s(x0))
        self.assertAlmostEqual(self.sigmoid.forward(x1), self.s(x1))

    # noinspection PyTypeChecker,PyArgumentList
    def test_sigmoid_propagate_tensor(self):

        """
        Test the forward method with tensor.
        """

        x = torch.FloatTensor([2.5, -2.5])
        res = self.sigmoid.forward(x)

        self.assertAlmostEqual(res[0], self.s(x[0]))
        self.assertAlmostEqual(res[1], self.s(x[1]))

    def test_sigmoid_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        # Test intercepting line

        bounds = torch.FloatTensor([[-2.5, -1]])
        relax = self.sigmoid.linear_relaxation(bounds[:, 0], bounds[:, 1])[1]
        gt_a = (self.s(bounds[0, 1]) - self.s(bounds[0, 0]))/(bounds[0, 1] - bounds[0, 0])
        gt_b = self.s(bounds[0, 1]) - (gt_a * bounds[0, 1])
        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

        # Test optimal tangent:

        bounds = torch.FloatTensor([[0, 2.5]])
        relax = self.sigmoid.linear_relaxation(bounds[:, 0], bounds[:, 1])[1]
        tangent_point = ((bounds[:, 1]**2 - bounds[:, 0]**2)/(2*(bounds[:, 1] - bounds[:, 0])))[0]
        gt_a = self.s(tangent_point) * (1-self.s(tangent_point))
        gt_b = self.s(tangent_point) - (gt_a * tangent_point)

        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

    def test_sigmoid_split_point(self):

        """
        Test the split_point() method.
        """

        split_point = self.sigmoid.split_point(torch.FloatTensor([-2.5]), torch.FloatTensor([2]))
        mid = (self.s(torch.FloatTensor([2])) + self.s(torch.FloatTensor([-2.5]))) / 2
        self.assertAlmostEqual(split_point, -torch.log((1 / mid) - 1))

    def test_tanh_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertTrue(self.tanh.is_monotonically_increasing)
        self.assertFalse(self.tanh.is_linear)

    def test_tanh_forward_float(self):

        """
        Test the forward method with floats.
        """

        x0, x1 = torch.FloatTensor([2.5]), torch.FloatTensor([-2.5])

        self.assertAlmostEqual(self.tanh.forward(x0), torch.tanh(x0))
        self.assertAlmostEqual(self.tanh.forward(x1), torch.tanh(x1))

    def test_tanh_forward_array(self):

        """
        Test the forward method with arrays.
        """

        x = torch.FloatTensor(([2.5, -2.5]))
        res = self.tanh.forward(x)

        self.assertAlmostEqual(res[0], torch.tanh(x[0]))
        self.assertAlmostEqual(res[1], torch.tanh(x[1]))

    def test_tanh_linear_relaxation(self):

        """
        Test the linear_relaxation() for positive, negative and mixed bounds.
        """

        # Test intercepting line

        bounds = torch.FloatTensor([[-2.5, -1]])
        relax = self.tanh.linear_relaxation(bounds[:, 0], bounds[:, 1])[1]
        gt_a = (torch.tanh(bounds[0, 1]) - torch.tanh(bounds[0, 0]))/(bounds[0, 1] - bounds[0, 0])
        gt_b = torch.tanh(bounds[0, 1]) - (gt_a * bounds[0, 1])
        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

        # Test optimal tangent:

        bounds = torch.FloatTensor([[0, 2.5]])
        relax = self.tanh.linear_relaxation(bounds[:, 0], bounds[:, 1])[1]
        tangent_point = ((bounds[:, 1]**2 - bounds[:, 0]**2)/(2*(bounds[:, 1] - bounds[:, 0])))[0]
        gt_a = 1 - torch.tanh(tangent_point)**2
        gt_b = torch.tanh(tangent_point) - (gt_a * tangent_point)

        self.assertAlmostEqual(relax[0, 0], gt_a)
        self.assertAlmostEqual(relax[0, 1], gt_b)

    def test_tanh_split_point(self):
        """
        Test the split_point() method.
        """

        split_point = self.tanh.split_point(torch.FloatTensor([-2.5]), torch.FloatTensor([2]))
        mid = (torch.tanh(torch.FloatTensor([2])) + torch.tanh(torch.FloatTensor([-2.5]))) / 2
        self.assertAlmostEqual(split_point, 0.5 * torch.log((1 + mid) / (1 - mid)))
