"""
Unittests for the ONNXParser class

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import torch
import unittest
import warnings
import logging

from verinet.util.config import CONFIG
CONFIG.LOGS_LEVEL = logging.ERROR

from verinet.parsers.onnx_parser import ONNXParser
from verinet.tests.simple_nn import SimpleNN, SimpleNNConv2d, SimpleNNBatchNorm2d, SimpleNNAvgPool2d, SimpleNNMean, \
    SimpleNNReshape, SimpleNNMulConstant, SimpleAddDynamic, SimpleDeepResidual


# noinspection PyCallingNonCallable
class TestONNXParser(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ImportWarning)

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
        Test saving and loading a simple 2-node networks with the ReLU op.
        """

        dummy = torch.rand((1, 2))
        self.simpleReLU.save(dummy, "test.onnx")
        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[1, 1]])

        gt = -4
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt))

    def testSigmoid(self):

        """
        Test saving and loading a simple 2-node networks with the Sigmoid op.
        """

        dummy_tensor = torch.Tensor([[-1, 10]])
        self.simpleSigmoid.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[1, 1]])

        gt = - (torch.sigmoid_(torch.Tensor([4]))) + torch.sigmoid_(torch.Tensor([1])) - 1
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt[0]))

    def testTanh(self):

        """
        Test saving and loading a simple 2-node networks with the Tanh op.
        """

        dummy_tensor = torch.Tensor([[-1, 10]])
        self.simpleTanh.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[1, 1]])

        gt = - (torch.tanh_(torch.Tensor([4]))) + torch.tanh_(torch.Tensor([1])) - 1
        res = model(x)[0][0]

        self.assertEqual(float(res), float(gt[0]))

    def testConv2d(self):

        """
        Test saving and loading a simple 2-node networks with the Conv2d op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleConv2d.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = 2
        res = model(x)[0][0, 0, 0, 0]
        self.assertEqual(float(res), float(gt))

    def testBatchNorm2d(self):

        """
        Test saving and loading a simple 2-node networks with the BatchNorm2d op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleBatchNorm2d.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        model.eval()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        std = torch.sqrt(torch.Tensor([2+1e-5]))
        gt = torch.sum((x + 0.5)/std * 0.5 + 0.5) - 1

        res = model(x)[0][0, 0, 0, 0]
        self.assertAlmostEqual(float(res), float(gt), places=5)

    def testAvgPool2d(self):

        """
        Test saving and loading a simple 2-node networks with the AvgPool2d op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleAvgPool2d.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([12/9, 3/9])
        res = model(x)[0][0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testMean(self):

        """
        Test saving and loading a simple 2-node networks with the Mean op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleMean.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([12/9, 3/9])
        res = model(x)[0][0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testReshape(self):

        """
        Test saving and loading a simple 2-node networks with the Reshape-op.
        """

        dummy_tensor = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
        self.simpleReshape.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]])

        gt = torch.Tensor([3])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testMulConstant(self):

        """
        Test saving and loading a simple 2-node networks with the Reshape-op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleMulConstant.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([6])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testAddDynamic(self):

        """
        Test saving and loading a simple 2-node networks with the AddDynamic op.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleAddDynamic.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([6])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))

    def testDeepResidual(self):

        """
        Test saving and loading a DeepResidual network.
        """

        dummy_tensor = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
        self.simpleDeepResidual.save(dummy_tensor, "test.onnx")

        parser = ONNXParser("test.onnx")
        os.remove("test.onnx")

        model = parser.to_pytorch()
        x = torch.Tensor([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

        gt = torch.Tensor([27*3])
        res = model(x)[0][0, 0, 0]

        self.assertAlmostEqual(float(res[0]), float(gt[0]))
