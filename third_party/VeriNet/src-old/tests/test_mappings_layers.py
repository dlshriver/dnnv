
"""
Unit-tests for the layers mappings

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as tf

from src.algorithm.mappings.layers import FC, Conv2d, BatchNorm2d


class TestMappingLayers(unittest.TestCase):

    def setUp(self):

        self.fc = FC()
        self.fc.params["weight"] = np.array([[1, 2, 3], [4, 5, 6]])

        self.fc.params["bias"] = np.array([1, 2])

        self.conv_2d = Conv2d()
        self.conv_2d.params["weight"] = np.array([[[[1, 1, 1], [1, 1, 0], [1, 0, 0]]]])
        self.conv_2d.params["bias"] = np.array([1])
        self.conv_2d.params["stride"] = (1, 1)
        self.conv_2d.params["padding"] = (1, 1)
        self.conv_2d.params["out_channels"] = 1
        self.conv_2d.params["kernel_size"] = (3, 3)
        self.conv_2d.params["in_shape"] = (1, 2, 2)

        self.batch_norm_2d = BatchNorm2d()
        self.batch_norm_2d.params["weight"] = np.array([2])
        self.batch_norm_2d.params["bias"] = np.array([1])
        self.batch_norm_2d.params["running_mean"] = np.array([1])
        self.batch_norm_2d.params["running_var"] = np.array([2])
        self.batch_norm_2d.params["in_shape"] = (1, 2, 2)
        self.batch_norm_2d.params["eps"] = 0.1

    def test_fc_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.fc.is_1d_to_1d)
        self.assertTrue(self.fc.is_linear)

    def test_fc_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3]]).T
        res = self.fc.propagate(x, add_bias=True)

        self.assertAlmostEqual(res[0, 0], -13)
        self.assertAlmostEqual(res[1, 0], -30)

    def test_conv_2d_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.conv_2d.is_1d_to_1d)
        self.assertTrue(self.conv_2d.is_linear)

    def test_conv_2d_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3, -4]]).T
        gt = np.array([[0, -5, -5, -9]]).T
        res = self.conv_2d.propagate(x, add_bias=True)

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

        gt = np.array([[-1, -6, -6, -10]]).T
        res = self.conv_2d.propagate(x, add_bias=False)

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

    def test_batch_norm_2d_properties(self):

        """
        Test the return values of is_linear() and is_1d_to_1d().
        """

        self.assertFalse(self.batch_norm_2d.is_1d_to_1d)
        self.assertTrue(self.batch_norm_2d.is_linear)

    def test_batch_norm_2d_propagate_array(self):

        """
        Test the propagate() method with arrays.
        """

        x = np.array([[-1, -2, -3, -4]]).T

        res = self.batch_norm_2d.propagate(x, add_bias=True)
        gt = tf.batch_norm(input=torch.Tensor(x),
                           running_mean=torch.Tensor(self.batch_norm_2d.params["running_mean"]),
                           running_var=torch.Tensor(self.batch_norm_2d.params["running_var"]),
                           weight=torch.Tensor(self.batch_norm_2d.params["weight"]),
                           bias=torch.Tensor(self.batch_norm_2d.params["bias"]),
                           eps=self.batch_norm_2d.params["eps"])

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)
