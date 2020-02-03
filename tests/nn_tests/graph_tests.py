import numpy as np
import unittest

from dnnv.nn import parse

from ..utils import network_artifact_dir as artifact_dir


class GraphTests(unittest.TestCase):
    def test_is_linear(self):
        dave_op_graph = parse(artifact_dir / "dave.onnx")
        self.assertTrue(dave_op_graph.is_linear)

        dronet_op_graph = parse(artifact_dir / "dronet.onnx")
        # falsified immediately at output layer
        self.assertFalse(dronet_op_graph.is_linear)
        # drop output layer, falsified slightly later
        self.assertFalse(dronet_op_graph[:-1, 0].is_linear)

    def test_slice(self):
        dave_op_graph = parse(artifact_dir / "dave.onnx")

        with self.assertRaises(TypeError) as cm:
            dave_op_graph["last_layer"]
        self.assertEqual(
            cm.exception.args[0], "Unsupported type for indexing operation graph: 'str'"
        )
        with self.assertRaises(ValueError) as cm:
            dave_op_graph[::2]
        self.assertEqual(
            cm.exception.args[0], "Slicing does not support non-unit steps."
        )
        with self.assertRaises(TypeError) as cm:
            dave_op_graph["start_layer", "last_layer"]
        self.assertEqual(
            cm.exception.args[0], "Unsupported type for slicing indices: 'str'"
        )
        with self.assertRaises(TypeError) as cm:
            dave_op_graph[0, 0, 0]
        self.assertEqual(
            cm.exception.args[0], "Unsupported indexing expression (0, 0, 0)"
        )
        with self.assertRaises(TypeError) as cm:
            dave_op_graph[:, "last_layer"]
        self.assertEqual(
            cm.exception.args[0], "Unsupported type for selecting operations: 'str'"
        )

        sliced_dave = dave_op_graph[0:None]
        for i in range(5):
            x = np.random.randn(1, 100, 100, 3).astype(np.float32)
            y1 = dave_op_graph(x).item()
            y2 = sliced_dave(x).item()
            self.assertAlmostEqual(y1, y2)

        dronet_op_graph = parse(artifact_dir / "dronet.onnx")
        sliced_dronet = dronet_op_graph[:-1][0]
        for i in range(5):
            x = np.random.randn(1, 200, 200, 1).astype(np.float32)
            y1 = dronet_op_graph(x)[:, 0].item()
            y2 = sliced_dronet(x).item()
            self.assertAlmostEqual(y1, y2)


if __name__ == "__main__":
    unittest.main()
