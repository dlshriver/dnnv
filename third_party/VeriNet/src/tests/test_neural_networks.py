"""
Unittests for the VeriNetNN classes

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import unittest
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.tests.simple_nn import SimpleNN, SimpleNNConv2d, SimpleNNBatchNorm2d, SimpleNNAvgPool2d, SimpleNNMean, \
    SimpleNNReshape, SimpleNNMulConstant, SimpleAddDynamic, SimpleDeepResidual


# noinspection PyCallingNonCallable
class TestVeriNetNN(unittest.TestCase):

    def setUp(self):

        self.simpleReLU = SimpleNN(activation="ReLU")
        self.simpleSigmoid = SimpleNN(activation="Sigmoid")
        self.simpleTanh = SimpleNN(activation="Tanh")

        self.simpleConv2d = SimpleNNConv2d()
        self.simpleBatchNorm2d = SimpleNNBatchNorm2d()
        self.simpleAvgPool2d = SimpleNNAvgPool2d()

        self.simpleMean = SimpleNNMean()
        self.simpleReshape = SimpleNNReshape()
        self.simpleMulConstant = SimpleNNMulConstant()
        self.simpleAddDynamic = SimpleAddDynamic()
        self.simpleDeepResidual = SimpleDeepResidual()

        self.simpleReLU.eval()
        self.simpleSigmoid.eval()
        self.simpleTanh.eval()

        self.simpleConv2d.eval()
        self.simpleBatchNorm2d.eval()
        self.simpleAvgPool2d.eval()

        self.simpleMean.eval()
        self.simpleReshape.eval()
        self.simpleMulConstant.eval()
        self.simpleAddDynamic.eval()
        self.simpleDeepResidual.eval()

    def testRelu(self):

        """
        Test a simple 2-node networks with the ReLU op.
        """

        model = self.simpleReLU
        x = torch.Tensor([[1, 1]])

        gt = -4
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt))

    def testSigmoid(self):

        """
        Test a simple 2-node networks with the Sigmoid op.
        """

        model = self.simpleSigmoid
        x = torch.Tensor([[1, 1]])

        gt = - (torch.sigmoid_(torch.Tensor([4]))) + torch.sigmoid_(torch.Tensor([1])) - 1
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt[0]))

    def testTanh(self):

        """
        Test a simple 2-node networks with the Tanh op.
        """

        model = self.simpleTanh
        x = torch.Tensor([[1, 1]])

        gt = - (torch.tanh_(torch.Tensor([4]))) + torch.tanh_(torch.Tensor([1])) - 1
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt[0]))

    def testConv2d(self):

        """
        Test a simple 2-node networks with the Conv2d op.
        """

        model = self.simpleConv2d
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = 2
        res = model(x)[0][0, 0, 0, 0]
        self.assertEqual(float(res), float(gt))

    def testBatchNorm2d(self):

        """
        Test a simple 2-node networks with the BatchNorm2d op.
        """

        model = self.simpleBatchNorm2d
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        std = torch.sqrt(torch.Tensor([2+1e-5]))
        gt = torch.sum((x + 0.5)/std * 0.5 + 0.5) - 1

        res = model(x)[0][0, 0, 0, 0]
        self.assertAlmostEqual(float(res), float(gt), places=5)

    def testAvgPool2d(self):

        """
        Test a simple 2-node networks with the AvgPool2d op.
        """

        model = self.simpleAvgPool2d
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([12/9, 3/9])
        res = model(x)[0][0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testMean(self):

        """
        Test a simple 2-node networks with the Mean op.
        """

        model = self.simpleMean
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([12/9, 3/9])
        res = model(x)[0][0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testReshape(self):

        """
        Test a simple 2-node networks with the Reshape-op.
        """

        model = self.simpleReshape
        x = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]])

        gt = torch.Tensor([3])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testMulConstant(self):

        """
        Test a simple 2-node networks with the Reshape-op.
        """

        model = self.simpleMulConstant
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([6])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testAddDynamic(self):

        """
        Test a simple 2-node networks with the AddDynamic op.
        """

        model = self.simpleAddDynamic
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([6])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testDeepResidual(self):

        """
        Test a DeepResidual network.
        """

        model = self.simpleDeepResidual
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([27*3])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))
