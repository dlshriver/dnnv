import numpy as np
import tensorflow as tf
import torch
import torchvision.models as models
import unittest

from ...utils import network_artifact_dir as artifact_dir
from dnnv.nn import parse

THRESHOLD = 1e-6


class KnownNetworkOutputTests(unittest.TestCase):
    def test_a_gt_b(self):
        op_graph = parse(artifact_dir / "a_gt_b.onnx")
        tf_model = op_graph.as_tf()
        for i in range(20):
            x = np.random.normal(size=(1, 2)).astype(np.float32)
            y = tf_model(x)
            assert (np.argmax(x, axis=1) == np.argmax(y, axis=1)).all()

    def test_const_one(self):
        op_graph = parse(artifact_dir / "const_one.onnx")
        tf_model = op_graph.as_tf()
        for i in range(20):
            x = np.random.normal(size=(1, 2)).astype(np.float32)
            y = tf_model(x)
            assert (y == 1).all()

    def test_const_zero(self):
        op_graph = parse(artifact_dir / "const_zero.onnx")
        tf_model = op_graph.as_tf()
        for i in range(20):
            x = np.random.normal(size=(1, 2)).astype(np.float32)
            y = tf_model(x)
            assert (y == 0).all()

    def test_sum_gt_one(self):
        op_graph = parse(artifact_dir / "sum_gt_one.onnx")
        tf_model = op_graph.as_tf()
        for i in range(10):
            x = np.random.normal(loc=0.0, size=(1, 10)).astype(np.float32)
            y = tf_model(x)
            x_sum = x.sum(axis=1) - 1
            y_expected = np.maximum(x_sum, np.zeros_like(x_sum))
            assert (np.abs(y - y_expected) < THRESHOLD).all()


class ImageNetTests(unittest.TestCase):
    def compare_model_output(self, pytorch_model, tf_model, threshold=THRESHOLD):
        pytorch_model.eval()
        for i in range(5):
            tf.compat.v1.reset_default_graph()
            x = torch.randn(1, 3, 224, 224)
            y_tf = tf_model(x.numpy())
            with torch.no_grad():
                y_pytorch = pytorch_model(x).numpy().reshape(y_tf.shape)
            self.assertTrue(
                (np.abs(y_tf - y_pytorch) < threshold).all(),
                f"{np.abs(y_tf - y_pytorch).max()} >= {threshold}",
            )


class Resnet34Tests(ImageNetTests):
    @classmethod
    def setUpClass(cls):
        cls.pytorch_model = models.resnet34(pretrained=True).eval()

    def test_resnet34__conv1__random_inputs(self):
        pytorch_model = torch.nn.Sequential(self.pytorch_model.conv1)
        model = parse(artifact_dir / "resnet34.onnx")[:2]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model)

    def test_resnet34__bn1__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1, self.pytorch_model.bn1
        )
        model = parse(artifact_dir / "resnet34.onnx")[:3]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model)

    def test_resnet34__relu__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1, self.pytorch_model.bn1, self.pytorch_model.relu
        )
        model = parse(artifact_dir / "resnet34.onnx")[:4]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model)

    def test_resnet34__maxpool__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1,
            self.pytorch_model.bn1,
            self.pytorch_model.relu,
            self.pytorch_model.maxpool,
        )
        model = parse(artifact_dir / "resnet34.onnx")[:5]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model)

    def test_resnet34__layer1block1__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1,
            self.pytorch_model.bn1,
            self.pytorch_model.relu,
            self.pytorch_model.maxpool,
            self.pytorch_model.layer1._modules["0"],
        )
        model = parse(artifact_dir / "resnet34.onnx")[:12]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 10)

    def test_resnet34__avgpool__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1,
            self.pytorch_model.bn1,
            self.pytorch_model.relu,
            self.pytorch_model.maxpool,
            self.pytorch_model.layer1,
            self.pytorch_model.layer2,
            self.pytorch_model.layer3,
            self.pytorch_model.layer4,
            self.pytorch_model.avgpool,
        )
        model = parse(artifact_dir / "resnet34.onnx")[:118]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 15)

    def test_resnet34__flatten__random_inputs(self):
        pytorch_model = torch.nn.Sequential(
            self.pytorch_model.conv1,
            self.pytorch_model.bn1,
            self.pytorch_model.relu,
            self.pytorch_model.maxpool,
            self.pytorch_model.layer1,
            self.pytorch_model.layer2,
            self.pytorch_model.layer3,
            self.pytorch_model.layer4,
            self.pytorch_model.avgpool,
        )
        model = parse(artifact_dir / "resnet34.onnx")[:119]
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 15)

    def test_resnet34_random_inputs(self):
        pytorch_model = self.pytorch_model
        model = parse(artifact_dir / "resnet34.onnx")
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 15)


class Resnet50Tests(ImageNetTests):
    @classmethod
    def setUpClass(cls):
        cls.pytorch_model = models.resnet50(pretrained=True).eval()

    def test_resnet50_random_inputs(self):
        pytorch_model = self.pytorch_model
        model = parse(artifact_dir / "resnet50.onnx")
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 10)


class Vgg16Tests(ImageNetTests):
    @classmethod
    def setUpClass(cls):
        cls.pytorch_model = models.vgg16(pretrained=True).eval()

    def test_vgg16_random_inputs(self):
        pytorch_model = self.pytorch_model
        model = parse(artifact_dir / "vgg16.onnx")
        tf_model = model.as_tf()
        self.compare_model_output(pytorch_model, tf_model, THRESHOLD * 10)


class DaveTests(unittest.TestCase):
    def test_dave(self):
        op_graph = parse(artifact_dir / "dave.onnx")
        tf_model_1 = op_graph.as_tf()
        tf_model_2 = op_graph[2:].simplify().as_tf()
        for i in range(5):
            x1 = np.random.randn(1, 100, 100, 3).astype(np.float32)
            y1 = tf_model_1(x1).item()
            x2 = x1.transpose((0, 3, 1, 2))
            y2 = tf_model_2(x2).item()
            self.assertAlmostEqual(y1, y2)


class DronetTests(unittest.TestCase):
    def test_dronet(self):
        op_graph = parse(artifact_dir / "dronet.onnx")
        tf_model_1 = op_graph.as_tf()
        tf_model_2 = op_graph[2:].simplify().as_tf()
        for i in range(5):
            x1 = np.random.randn(1, 200, 200, 1).astype(np.float32)
            y1 = tf_model_1(x1)
            x2 = x1.transpose((0, 3, 1, 2))
            y2 = tf_model_2(x2)
            self.assertEqual(y1.shape, y2.shape)
            self.assertEqual(y1.shape[0], 1)
            self.assertAlmostEqual(y1[0, 0], y2[0, 0], places=6)
            self.assertAlmostEqual(y1[0, 1], y2[0, 1], places=6)


if __name__ == "__main__":
    unittest.main()
