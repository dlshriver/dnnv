
"""
Unittests for the linear operations

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import unittest
import logging

import torch
import torch.nn as nn

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.sip_torch.operations.linear import Identity, Reshape, Flatten, FC, Conv2d, \
    AvgPool2d, Mean, MulConstant, AddDynamic, Crop, Transpose, AddConstant
from verinet.neural_networks.custom_layers import Reshape as TorchReshape
from verinet.neural_networks.custom_layers import Mean as TorchMean
from verinet.neural_networks.custom_layers import Crop as TorchCrop
from verinet.neural_networks.custom_layers import MulConstant as TorchMulConstant
from verinet.neural_networks.custom_layers import AddDynamic as TorchAddDynamic
from verinet.neural_networks.custom_layers import Transpose as TorchTranspose


# noinspection PyCallingNonCallable,PyTypeChecker
class TestOperationsLinear(unittest.TestCase):

    # noinspection PyArgumentList
    def setUp(self):

        self.identity = Identity()
        self.flatten = Flatten()

        self.reshape = Reshape()
        self.reshape.params["shape"] = (1, 3, 3, 1)

        self.transpose = Transpose()
        self.transpose.params["dim_order"] = (0, 3, 1, 2)
        self.transpose.params["in_shape"] = torch.LongTensor((1, 2, 3))

        self.fc = FC()
        self.fc.params["weight"] = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        self.fc.params["bias"] = torch.Tensor([1, 2])

        self.conv_2d = Conv2d()
        self.conv_2d.params["weight"] = torch.Tensor([[[[1, 1, 1], [1, 1, 0], [1, 0, 0]]]])
        self.conv_2d.params["bias"] = torch.Tensor([1])
        self.conv_2d.params["stride"] = (1, 1)
        self.conv_2d.params["padding"] = (1, 1)
        self.conv_2d.params["out_channels"] = 1
        self.conv_2d.params["kernel_size"] = (3, 3)
        self.conv_2d.params["in_shape"] = torch.LongTensor((1, 2, 2))
        self.conv_2d.params["groups"] = 1

        self.avg_pool_2d = AvgPool2d()
        self.avg_pool_2d.params["stride"] = (1, 1)
        self.avg_pool_2d.params["padding"] = (0, 0)
        self.avg_pool_2d.params["kernel_size"] = (2, 2)
        self.avg_pool_2d.params["in_shape"] = (1, 3, 3)

        self.mean = Mean()
        self.mean.params["dims"] = (2, 3)
        self.mean.params["keepdim"] = False
        self.mean.params["in_shape"] = (1, 2, 2)

        self.mul_constant = MulConstant()
        self.mul_constant.params["multiplier"] = torch.Tensor((2,))
        self.mul_constant.params["in_shape"] = (3, 4, 4)

        self.add_constant = AddConstant()
        self.add_constant.params["term"] = torch.Tensor((2,))
        self.add_constant.params["in_shape"] = (3, 4, 4)

        self.add_dynamic = AddDynamic()
        self.add_dynamic.params["in_shape"] = (3, 4, 4)

        self.crop = Crop()
        self.crop.params["crop"] = 1
        self.crop.params["in_shape"] = (3, 4, 4)

    def test_id_properties(self):

        """
        Test the properties of the id-node.
        """

        self.assertTrue(self.identity.is_monotonically_increasing)
        self.assertTrue(self.identity.is_linear)
        self.assertTrue(len(self.identity.required_params) == 0)
        self.assertTrue(nn.Identity in self.identity.abstracted_torch_funcs())

    # noinspection PyTypeChecker,PyArgumentList
    def test_id_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.identity.out_shape(in_shape=torch.LongTensor((32, 32, 3)))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], 32)
        self.assertEqual(out_shape[2], 3)

    # noinspection PyArgumentList,PyTypeChecker
    def test_id_forward(self):

        """
        Test the forward method.
        """

        x = torch.Tensor([[-1, -2, -3]])
        res = self.identity.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_id_ssip_forward(self):

        """
        Tests the ssip_forward method.
        """

        x = torch.Tensor([[[-1, -2, -3]]])
        res = self.identity.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 3)

    def test_id_rsip_backward(self):

        """
        Tests the rsip_backward method.
        """

        x = torch.Tensor([[-1, -2, -3]])
        res = self.identity.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_flatten_properties(self):

        """
        Test the properties of the flatten-node.
        """

        self.assertTrue(self.flatten.is_monotonically_increasing)
        self.assertTrue(self.flatten.is_linear)
        self.assertTrue(len(self.flatten.required_params) == 0)
        self.assertTrue(nn.Flatten in self.flatten.abstracted_torch_funcs())

    # noinspection PyTypeChecker,PyArgumentList
    def test_flatten_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.flatten.out_shape(in_shape=torch.Tensor((32, 32, 3)))
        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 1)
        self.assertEqual(out_shape[0], 32*32*3)

    # noinspection PyArgumentList,PyTypeChecker
    def test_flatten_forward(self):

        """
        Test the forward method.
        """

        x = torch.FloatTensor([[-1, -2, -3]])
        res = self.flatten.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_flatten_ssip_forward(self):

        """
        Tests the ssip_forward method
        """

        x = torch.FloatTensor([[[-1, -2, -3]]])
        res = self.flatten.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 3)

    def test_flatten_rsip_backward(self):

        """
        Tests the rsip_backward method.
        """

        x = torch.FloatTensor([[-1, -2, -3]])
        res = self.flatten.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_reshape_properties(self):

        """
        Test the properties of the reshape-node.
        """

        self.assertTrue(self.reshape.is_monotonically_increasing)
        self.assertTrue(self.reshape.is_linear)
        self.assertTrue(len(self.reshape.required_params) == 1)
        self.assertTrue("shape" in self.reshape.required_params)
        self.assertTrue(TorchReshape in self.reshape.abstracted_torch_funcs())

    # noinspection PyTypeChecker,PyArgumentList
    def test_reshape_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.reshape.out_shape(in_shape=torch.Tensor((1, 9, )))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))

        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 3)
        self.assertEqual(out_shape[1], 3)
        self.assertEqual(out_shape[2], 1)

    # noinspection PyArgumentList,PyTypeChecker
    def test_reshape_forward(self):

        """
        Test the forward method.
        """

        x = torch.FloatTensor([[-1, -2, -3]])
        res = self.reshape.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_reshape_ssip_forward(self):

        """
        Tests the ssip_forward method.
        """

        x = torch.FloatTensor([[[-1, -2, -3]]])
        res = self.reshape.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 3)

    def test_reshape_rsip_backward(self):

        """
        Tests the rsip_backward method.
        """

        x = torch.FloatTensor([[-1, -2, -3]])
        res = self.reshape.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_fc_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertFalse(self.fc.is_monotonically_increasing)
        self.assertTrue(self.fc.is_linear)

        for param in ["weight", "bias"]:
            self.assertTrue(param in self.fc.required_params)

        self.assertTrue(nn.modules.linear.Linear in self.fc.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_fc_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.fc.out_shape(in_shape=None)

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 1)
        self.assertTrue(out_shape[0] == 2)

    # noinspection PyArgumentList,PyTypeChecker
    def test_fc_forward(self):

        """
        Test the forward method.
        """

        x = torch.FloatTensor([[-1, -2, -3]]).T
        res = self.fc.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0], - 13)
        self.assertAlmostEqual(res[1, 0], - 30)

    def test_fc_ssip_forward(self):

        """
        Tests the ssip_forward method.
        """

        self.fc.params["weight"] = torch.Tensor([[-1, 1]]).T
        self.fc.params["bias"] = torch.Tensor([0, 1])

        x = torch.FloatTensor([[[-1, -2, -3]], [[2, 3, 4]]])

        res = self.fc.ssip_forward([x])

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 3)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 4)
        self.assertAlmostEqual(res[0, 1, 0].numpy(), -1)
        self.assertAlmostEqual(res[0, 1, 1].numpy(), -2)
        self.assertAlmostEqual(res[0, 1, 2].numpy(), -2)

        self.assertAlmostEqual(res[1, 0, 0].numpy(), 1)
        self.assertAlmostEqual(res[1, 0, 1].numpy(), 2)
        self.assertAlmostEqual(res[1, 0, 2].numpy(), 3)
        self.assertAlmostEqual(res[1, 1, 0].numpy(), 2)
        self.assertAlmostEqual(res[1, 1, 1].numpy(), 3)
        self.assertAlmostEqual(res[1, 1, 2].numpy(), 5)

    def test_fc_rsip_backward(self):

        """
        Test the forward method.
        """

        x = torch.FloatTensor([[1, 2, 3], [2, 0, -1]])
        weight = torch.FloatTensor([[1, 2, 3], [2, 3, 4]])
        bias = torch.FloatTensor([0, 1])

        fc = FC()
        fc.params = {"weight": weight, "bias": bias}

        gt = torch.FloatTensor([[5, 8, 11, 5], [2, 4, 6, -1]])

        res = fc.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):

                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_conv_2d_properties(self):

        """
        Test the properties of the convolutional node.
        """

        self.assertFalse(self.conv_2d.is_monotonically_increasing)
        self.assertTrue(self.conv_2d.is_linear)

        for param in ["weight", "bias", "kernel_size", "padding", "stride", "in_channels", "out_channels"]:
            self.assertTrue(param in self.conv_2d.required_params)

        self.assertTrue(nn.Conv2d in self.conv_2d.abstracted_torch_funcs())

    # noinspection PyArgumentList,PyTypeChecker
    def test_conv_2d_forward(self):

        """
        Test the forward method with arrays.
        """

        x = torch.FloatTensor([[-1, -2, -3, -4]]).T
        gt = torch.FloatTensor([[0, -5, -5, -9]]).T
        res = self.conv_2d.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

        gt = torch.Tensor([[-1, -6, -6, -10]]).T
        res = self.conv_2d.forward(x, add_bias=False)

        for i, val in enumerate(gt[:, 0]):
            self.assertAlmostEqual(res[i, 0], val)

    def test_conv2d_ssip_forward(self):

        """
        Tests the ssip_forward method.
        """

        x = torch.FloatTensor([[[1, 1, 1], [2, 2, 2], [3, 3, 3],
                                [4, 4, 4], [5, 5, 5], [6, 6, 6],
                                [7, 7, 7], [8, 8, 8], [9, 9, 9]],
                               [[1, 1, 1], [2, 2, 2], [3, 3, 3],
                                [4, 4, 4], [5, 5, 5], [6, 6, 6],
                                [7, 7, 7], [8, 8, 8], [9, 9, 9]]])

        weight = torch.FloatTensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        bias = torch.FloatTensor([0])

        conv_2d = Conv2d()
        conv_2d.params["weight"] = weight
        conv_2d.params["bias"] = bias
        conv_2d.params["stride"] = (1, 1)
        conv_2d.params["padding"] = (1, 1)
        conv_2d.params["out_channels"] = 1
        conv_2d.params["kernel_size"] = (3, 3)
        conv_2d.params["in_shape"] = torch.LongTensor((1, 3, 3))
        conv_2d.params["groups"] = 1

        gt = torch.FloatTensor([[[6, 6, 6], [8, 8, 8], [3, 3, 3],
                                 [12, 12, 12], [15, 15, 15], [8, 8, 8],
                                 [7, 7, 7], [12, 12, 12], [14, 14, 14]],
                                [[6, 6, 6], [8, 8, 8], [3, 3, 3],
                                 [12, 12, 12], [15, 15, 15], [8, 8, 8],
                                 [7, 7, 7], [12, 12, 12], [14, 14, 14]]])

        res = conv_2d.ssip_forward([x])
        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    self.assertAlmostEqual(float(res[i, j, k]), float(gt[i, j, k]))

    def test_conv2d_rsip_backward(self):

        """
        Test the rsip_backward method.
        """

        x = torch.FloatTensor([[1, 0, 0, 0, 0],
                               [2, 3, 4, 5, 6]
                               ])

        weight = torch.FloatTensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]])
        bias = torch.FloatTensor([1])

        conv_2d = Conv2d()
        conv_2d.params["weight"] = weight
        conv_2d.params["bias"] = bias
        conv_2d.params["stride"] = (2, 2)
        conv_2d.params["padding"] = (1, 1)
        conv_2d.params["out_channels"] = 1
        conv_2d.params["kernel_size"] = (3, 3)
        conv_2d.params["in_shape"] = torch.LongTensor((1, 3, 3))
        conv_2d.params["groups"] = 1

        gt = torch.FloatTensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1], [2, 0, 3, 0, 14, 0, 4, 0, 5, 20]])

        res = conv_2d.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):

                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_conv2d_rsip_backward_2_channels(self):

        """
        Test the rsip_backward method with 2 input and output channels
        """

        x = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 3, 4, 5, 6, 7, 8, 9, 10]
                               ])

        weight = torch.FloatTensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]],
                                    [[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]]])
        bias = torch.FloatTensor([1, 2])

        conv_2d = Conv2d()
        conv_2d.params["weight"] = weight
        conv_2d.params["bias"] = bias
        conv_2d.params["stride"] = (2, 2)
        conv_2d.params["padding"] = (1, 1)
        conv_2d.params["out_channels"] = 2
        conv_2d.params["kernel_size"] = (3, 3)
        conv_2d.params["in_shape"] = torch.LongTensor((2, 3, 3))
        conv_2d.params["groups"] = 1

        gt = torch.FloatTensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                [2, 13, 3, 14, 14, 16, 4, 17, 5, 2, 13, 3, 14, 14, 16, 4, 17, 5, 84]])

        res = conv_2d.rsip_backward(x, add_bias=True)[0]
        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):

                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_conv2d_rsip_backward_groups(self):

        """
        Test the rsip_backward method with groups in the convolution
        """

        x = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 3, 4, 5, 6, 7, 8, 9, 10]
                               ])

        weight = torch.FloatTensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]],
                                    [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]])
        bias = torch.FloatTensor([1, 2])

        conv_2d = Conv2d()
        conv_2d.params["weight"] = weight
        conv_2d.params["bias"] = bias
        conv_2d.params["stride"] = (2, 2)
        conv_2d.params["padding"] = (1, 1)
        conv_2d.params["out_channels"] = 2
        conv_2d.params["kernel_size"] = (3, 3)
        conv_2d.params["in_shape"] = torch.LongTensor((2, 3, 3))
        conv_2d.params["groups"] = 2

        gt = torch.FloatTensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [2, 0, 3, 0, 14, 0, 4, 0, 5, 0, 13, 0, 14, 0, 16, 0, 17, 0, 84]])

        res = conv_2d.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):

                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_conv2d_rsip_backward_output_padding(self):

        """
        Test the rsip_backward method with 2 input and output channels
        """

        x = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 2, 2, 2, 1, 1, 1, 1, 10]
                               ])

        weight = torch.FloatTensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                                    [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]])
        bias = torch.FloatTensor([1, 2])

        conv_2d = Conv2d()
        conv_2d.params["weight"] = weight
        conv_2d.params["bias"] = bias
        conv_2d.params["stride"] = (2, 2)
        conv_2d.params["padding"] = (0, 0)
        conv_2d.params["out_channels"] = 2
        conv_2d.params["kernel_size"] = (2, 2)
        conv_2d.params["in_shape"] = torch.LongTensor((2, 5, 5))
        conv_2d.params["groups"] = 1

        gt = torch.FloatTensor([[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0,
                                 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 26]])

        res = conv_2d.rsip_backward(x, add_bias=True)[0]
        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):

                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_avg_pool_2d_properties(self):

        """
        Test the properties of the convolutional node.
        """

        self.assertFalse(self.avg_pool_2d.is_monotonically_increasing)
        self.assertTrue(self.avg_pool_2d.is_linear)

        for param in ["kernel_size", "padding", "stride"]:
            self.assertTrue(param in self.avg_pool_2d.required_params)

        self.assertTrue(nn.AvgPool2d in self.avg_pool_2d.abstracted_torch_funcs())

    # noinspection PyArgumentList,PyTypeChecker
    def test_avg_pool_2d_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        x = torch.FloatTensor([[[1, 2, 3],
                                [2, 3, 4],
                                [3, 4, 5],
                                [4, 5, 6],
                                [5, 6, 7],
                                [6, 7, 8],
                                [7, 8, 9],
                                [8, 9, 10],
                                [9, 10, 11]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]]
                              )

        res = self.avg_pool_2d.ssip_forward([x])
        self.assertTrue(isinstance(res, torch.FloatTensor))

        y = torch.FloatTensor([[[3, 4, 5],
                                [4, 5, 6],
                                [6, 7, 8],
                                [7, 8, 9]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]]
                              )

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                self.assertAlmostEqual(res[0, i, j], y[0, i, j])

    def test_avg_pool_2d_rsip_rsip_backward_2_channels(self):

        """
        Test the rsip_backward method with 2 input and output channels
        """

        x = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 2, 2, 3, 3, 4, 4, 10]
                               ])

        avg_pool_2d = AvgPool2d()
        avg_pool_2d.params["stride"] = (2, 2)
        avg_pool_2d.params["padding"] = (1, 1)
        avg_pool_2d.params["kernel_size"] = (3, 3)
        avg_pool_2d.params["in_shape"] = torch.LongTensor((2, 3, 3))

        gt = torch.FloatTensor([[1/9, 1/9, 0, 1/9, 1/9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1/9, 2/9, 1/9, 3/9, 6/9, 3/9, 2/9, 4/9, 2/9,
                                 3/9, 6/9, 3/9, 7/9, 14/9, 7/9, 4/9, 8/9, 4/9, 10]])

        res = avg_pool_2d.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_mean_properties(self):

        """
        Test the properties of the convolutional node.
        """

        self.assertFalse(self.mean.is_monotonically_increasing)
        self.assertTrue(self.mean.is_linear)

        for param in ["dims", "keepdim"]:
            self.assertTrue(param in self.mean.required_params)

        self.assertTrue(TorchMean in self.mean.abstracted_torch_funcs())

    # noinspection PyArgumentList,PyTypeChecker
    def test_mean_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        x = torch.FloatTensor([[[1, 2, 3],
                                [2, 3, 4],
                                [3, 4, 5],
                                [4, 5, 6]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]]
                              )

        res = self.mean.ssip_forward([x])
        self.assertTrue(isinstance(res, torch.FloatTensor))

        y = torch.FloatTensor([[[2.5, 3.5, 4.5]],
                               [[0, 0, 0]]]
                              )

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                self.assertAlmostEqual(res[0, i, j], y[0, i, j])

    def test_mean_rsip_backward_2_channels(self):

        """
        Test the rsip_backward method with 2 input and output channels
        """

        x = torch.FloatTensor([[2, 4, 0],
                               [9, 18, 10]
                               ])

        mean = Mean()
        mean.params["dims"] = (2, 3)
        mean.params["in_shape"] = torch.LongTensor((2, 3, 3))

        gt = torch.FloatTensor([[2/9, 2/9, 2/9, 2/9, 2/9, 2/9, 2/9, 2/9, 2/9,
                                 4/9, 4/9, 4/9, 4/9, 4/9, 4/9, 4/9, 4/9, 4/9, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 10]])

        res = mean.rsip_backward(x)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_crop_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertTrue(self.crop.is_monotonically_increasing)
        self.assertTrue(self.crop.is_linear)

        for param in ["crop"]:
            self.assertTrue(param in self.crop.required_params)

        self.assertTrue(TorchCrop in self.crop.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_crop_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.crop.out_shape(in_shape=(3, 4, 4))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 3)
        self.assertEqual(out_shape[1], 2)
        self.assertEqual(out_shape[2], 2)

    # noinspection PyArgumentList,PyTypeChecker
    def test_crop_forward(self):

        """
        Test the forward method.
        """

        x = torch.ones((1, 3*4*4))
        res = self.crop.forward(x)

        self.assertTrue(isinstance(res, torch.FloatTensor))
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1], 12)

        for i in range(1):
            for j in range(12):
                self.assertEqual(res[i, j], x[i, j])

    def test_crop_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        x = torch.ones((2, 1, 3*4*4 + 1))
        x[1, :] += 1

        res = self.crop.ssip_forward([x])
        gt = torch.FloatTensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]])

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[1]):
                    self.assertAlmostEqual(float(res[i, j, k]), float(gt[i, j, k]))

    def test_crop_rsip_backward_backward(self):

        """
        Test the rsip_backward method.
        """

        x = torch.ones((1, 3*2*2 + 1))
        res = self.crop.rsip_backward(x)[0]

        gt = torch.FloatTensor([[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                 1]])

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_mul_constant_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertFalse(self.mul_constant.is_monotonically_increasing)
        self.assertTrue(self.mul_constant.is_linear)

        for param in ["multiplier"]:
            self.assertTrue(param in self.mul_constant.required_params)

        self.assertTrue(TorchMulConstant in self.mul_constant.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_mul_constant_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.mul_constant.out_shape(in_shape=torch.LongTensor((32, 32, 3)))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], 32)
        self.assertEqual(out_shape[2], 3)

    # noinspection PyArgumentList,PyTypeChecker
    def test_mul_constant_forward(self):

        """
        Test the forward method.
        """

        self.mul_constant.params["in_shape"] = (1,)
        x = torch.Tensor([[-1, -2, -3]])
        res = self.mul_constant.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 2)
        self.assertAlmostEqual(res[0, 1].numpy(), - 4)
        self.assertAlmostEqual(res[0, 2].numpy(), - 6)

    def test_mul_constant_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        self.mul_constant.params["in_shape"] = (1,)

        x = torch.Tensor([[[-1, -2, -3]], [[1, 2, 3]]])
        res = self.mul_constant.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 4)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 6)

        self.assertAlmostEqual(res[1, 0, 0].numpy(), 2)
        self.assertAlmostEqual(res[1, 0, 1].numpy(), 4)
        self.assertAlmostEqual(res[1, 0, 2].numpy(), 6)

    def test_mul_constant_ssip_forward_neg(self):

        """
        Test the ssip_forward method.
        """

        self.mul_constant.params["in_shape"] = torch.LongTensor((1,))
        self.mul_constant.params["multiplier"] = torch.Tensor((-2,))

        x = torch.Tensor([[[-1, -2, -3]], [[2, 3, 4]]])
        res = self.mul_constant.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 4)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 6)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 8)

        self.assertAlmostEqual(res[1, 0, 0].numpy(), 2)
        self.assertAlmostEqual(res[1, 0, 1].numpy(), 4)
        self.assertAlmostEqual(res[1, 0, 2].numpy(), 6)

    def test_mul_constant_rsip_backward_backward(self):

        """
        Test the rsip_backward method.
        """

        self.mul_constant.params["in_shape"] = (2, )

        x = torch.Tensor([[-1, -2, -3]])
        res = self.mul_constant.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 2)
        self.assertAlmostEqual(res[0, 1].numpy(), - 4)
        self.assertAlmostEqual(res[0, 2].numpy(), - 3)

    def test_add_constant_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertTrue(self.add_constant.is_monotonically_increasing)
        self.assertTrue(self.add_constant.is_linear)

        for param in ["term"]:
            self.assertTrue(param in self.add_constant.required_params)

        self.assertTrue(TorchMulConstant in self.mul_constant.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_add_constant_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.mul_constant.out_shape(in_shape=torch.LongTensor((32, 32, 3)))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], 32)
        self.assertEqual(out_shape[2], 3)

    # noinspection PyArgumentList,PyTypeChecker
    def test_add_constant_forward(self):

        """
        Test the forward method.
        """

        self.add_constant.params["in_shape"] = (1,)
        x = torch.Tensor([[-1, -2, -3]])
        res = self.add_constant.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), 1)
        self.assertAlmostEqual(res[0, 1].numpy(), 0)
        self.assertAlmostEqual(res[0, 2].numpy(), -1)

    def test_add_constant_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        self.add_constant.params["in_shape"] = (1,)

        x = torch.Tensor([[[-1, -2, -3]], [[1, 2, 3]]])
        res = self.add_constant.ssip_forward([x], add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 1)

        self.assertAlmostEqual(res[1, 0, 0].numpy(), 1)
        self.assertAlmostEqual(res[1, 0, 1].numpy(), 2)
        self.assertAlmostEqual(res[1, 0, 2].numpy(), 5)

    def test_add_constant_rsip_backward_backward(self):

        """
        Test the rsip_backward method.
        """

        self.add_constant.params["in_shape"] = (2, )

        x = torch.Tensor([[-1, -2, -3]])
        res = self.add_constant.rsip_backward(x, add_bias=True)[0]

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0, 2].numpy(), - 9)

    def test_add_dynamic_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertTrue(self.add_dynamic.is_monotonically_increasing)
        self.assertTrue(self.add_dynamic.is_linear)
        self.assertTrue(TorchAddDynamic in self.add_dynamic.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_add_dynamic_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.add_dynamic.out_shape(in_shape=torch.LongTensor((32, 32, 3)))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], 32)
        self.assertEqual(out_shape[2], 3)

    # noinspection PyArgumentList,PyTypeChecker
    def test_add_dynamic_forward(self):

        """
        Test the forward method.
        """

        x = [torch.Tensor([[-1, -2, -3]]), torch.Tensor([[-2, -3, -4]])]
        res = self.add_dynamic.forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0].numpy(), - 3)
        self.assertAlmostEqual(res[0, 1].numpy(), - 5)
        self.assertAlmostEqual(res[0, 2].numpy(), - 7)

    def test_add_dynamic_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        x = [torch.Tensor([[[-1, -2, -3]]]), torch.Tensor([[[-2, -3, -4]]])]
        res = self.add_dynamic.ssip_forward(x, add_bias=True)

        self.assertTrue(isinstance(res, torch.FloatTensor))

        self.assertAlmostEqual(res[0, 0, 0].numpy(), - 3)
        self.assertAlmostEqual(res[0, 0, 1].numpy(), - 5)
        self.assertAlmostEqual(res[0, 0, 2].numpy(), - 7)

    def test_add_dynamic_rsip_backward_backward(self):

        """
        Test the rsip_backward method.
        """

        x = torch.Tensor([[-1, -2, -3]])
        res = self.add_dynamic.rsip_backward(x, add_bias=True)

        self.assertTrue(isinstance(res, list))

        self.assertAlmostEqual(res[0][0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[0][0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[0][0, 2].numpy(), - 3)
        self.assertAlmostEqual(res[1][0, 0].numpy(), - 1)
        self.assertAlmostEqual(res[1][0, 1].numpy(), - 2)
        self.assertAlmostEqual(res[1][0, 2].numpy(), 0)

    def test_transpose_properties(self):

        """
        Test the properties of the operation.
        """

        self.assertTrue(self.transpose.is_monotonically_increasing)
        self.assertTrue(self.transpose.is_linear)
        self.assertTrue(TorchTranspose in self.transpose.abstracted_torch_funcs())

    # noinspection PyTypeChecker
    def test_transpose_out_shape(self):

        """
        Test that the type and value of out_shape are correct.
        """

        out_shape = self.transpose.out_shape(in_shape=torch.LongTensor((32, 32, 3)))

        self.assertTrue(isinstance(out_shape, torch.LongTensor))
        self.assertTrue(len(out_shape) == 3)
        self.assertEqual(out_shape[0], 3)
        self.assertEqual(out_shape[1], 32)
        self.assertEqual(out_shape[2], 32)

    # noinspection PyArgumentList,PyTypeChecker
    def test_transpose_forward(self):

        """
        Test the forward method.
        """

        x = [torch.Tensor([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5],
                           [5, 6],
                           [6, 7],
                           ])]

        res = self.transpose.forward(x, add_bias=True)
        gt = torch.Tensor([[1, 2],
                           [4, 5],
                           [2, 3],
                           [5, 6],
                           [3, 4],
                           [6, 7],
                           ])

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(res[i, j]), float(gt[i, j]))

    def test_transpose_ssip_forward(self):

        """
        Test the ssip_forward method.
        """

        x = [torch.Tensor([[[1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7]],
                           [[-1, -2],
                            [-2, -3],
                            [-3, -4],
                            [-4, -5],
                            [-5, -6],
                            [-6, -7]]])]

        res = self.transpose.ssip_forward(x, add_bias=True)
        gt = torch.Tensor([[[1, 2],
                            [4, 5],
                            [2, 3],
                            [5, 6],
                            [3, 4],
                            [6, 7]],
                           [[-1, -2],
                            [-4, -5],
                            [-2, -3],
                            [-5, -6],
                            [-3, -4],
                            [-6, -7]]])

        self.assertTrue(isinstance(res, torch.FloatTensor))

        for k in range(2):
            for i in range(gt.shape[1]):
                for j in range(gt.shape[2]):
                    self.assertAlmostEqual(float(res[k, i, j]), float(gt[k, i, j]))

    def test_transpose_rsip_backward_backward(self):

        """
        Test the rsip_backward method.
        """

        x = torch.Tensor([[1, 4, 2, 5, 3, 6, 7],
                          [2, 5, 3, 6, 4, 7, 8]])

        res = self.transpose.rsip_backward(x, add_bias=True)

        gt = torch.Tensor([[1, 2, 3, 4, 5, 6, 7],
                           [2, 3, 4, 5, 6, 7, 8]])

        self.assertTrue(isinstance(res[0], torch.FloatTensor))

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                self.assertAlmostEqual(float(res[0][i, j]), float(gt[i, j]))
