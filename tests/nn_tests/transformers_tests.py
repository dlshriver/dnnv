import numpy as np
import tensorflow as tf
import unittest

from dnnv.nn import parse, OperationGraph, operations
from dnnv.nn.transformers import DropPrefix
from ..utils import network_artifact_dir as artifact_dir

THRESHOLD = 1e-6


class AssertCloseMixin:
    def assert_close(
        self,
        model1,
        model2,
        threshold=THRESHOLD,
        input_shape=(1, 3, 224, 224),
        loc=0.0,
        scale=1.0,
    ):
        for i in range(5):
            tf.compat.v1.reset_default_graph()
            x = np.random.normal(loc=loc, scale=scale, size=input_shape).astype(
                np.float32
            )
            y_1 = model1(x)
            y_2 = model2(x)
            self.assertTrue(
                (np.abs(y_1 - y_2) < threshold).all(),
                f"{np.abs(y_1 - y_2).max()} >= {threshold}",
            )


class DropPrefixTests(unittest.TestCase):
    def test_sum_gt_one(self):
        op_graph = parse(artifact_dir / "sum_gt_one.onnx")
        output_op = op_graph.output_operations[0]  # gemm
        prefix = OperationGraph([output_op.a])
        op_graph_suffix = OperationGraph(op_graph.walk(DropPrefix(prefix)))
        x = np.ones((1, 1)).astype(np.float32)
        y = op_graph_suffix(x)
        y_expected = np.ones((1, 1))
        self.assertTrue((np.abs(y_expected - y) <= THRESHOLD).all())
        x = np.random.normal(size=(10, 1)).astype(np.float32)
        y = op_graph_suffix(x)
        y_expected = x.copy()
        self.assertTrue((np.abs(y_expected - y) <= THRESHOLD).all())

        prefix_output_op = prefix.output_operations[0]  # relu
        prefix = OperationGraph([prefix_output_op.x])
        op_graph_suffix = OperationGraph(op_graph.walk(DropPrefix(prefix)))
        x = np.ones((1, 1)).astype(np.float32)
        y = op_graph_suffix(x)
        y_expected = np.ones((1, 1))
        self.assertTrue((np.abs(y_expected - y) <= THRESHOLD).all())
        x = np.random.normal(size=(10, 1)).astype(np.float32)
        y = op_graph_suffix(x)
        y_expected = np.clip(x, 0.0, None)
        self.assertTrue((np.abs(y_expected - y) <= THRESHOLD).all())

    def test_a_gt_b(self):
        op_graph = parse(artifact_dir / "a_gt_b.onnx")
        output_op = op_graph.output_operations[0]  # softmax
        prefix = OperationGraph([output_op.x])
        op_graph_suffix = OperationGraph(op_graph.walk(DropPrefix(prefix)))
        x = np.random.normal(size=(10, 2)).astype(np.float32)
        y = op_graph_suffix(x)
        a_gt_b = x[:, 0] > x[:, 1]
        y_0_gt_1 = y[:, 0] > y[:, 1]
        self.assertTrue(np.all(~(a_gt_b ^ y_0_gt_1)))

        prefix_output_op = prefix.output_operations[0]  # gemm
        prefix = OperationGraph([prefix_output_op.a])
        op_graph_suffix = OperationGraph(op_graph.walk(DropPrefix(prefix)))
        x = np.clip(np.random.normal(size=(10, 2)), 0.0, None).astype(np.float32)
        y = op_graph_suffix(x)
        a_gt_b = x[:, 0] > x[:, 1]
        y_0_gt_1 = y[:, 0] > y[:, 1]
        self.assertTrue(np.all(~(a_gt_b ^ y_0_gt_1)))


class SimplifyTests(AssertCloseMixin, unittest.TestCase):
    def test_a_gt_b(self):
        op_graph = parse(artifact_dir / "a_gt_b.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 2))

    def test_const_one(self):
        op_graph = parse(artifact_dir / "const_one.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 2))

    def test_const_zero(self):
        op_graph = parse(artifact_dir / "const_zero.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 2))

    def test_sum_gt_one(self):
        op_graph = parse(artifact_dir / "sum_gt_one.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 10))

    def test_resnet34(self):
        op_graph = parse(artifact_dir / "resnet34.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, threshold=10 * THRESHOLD)

    def test_resnet50(self):
        op_graph = parse(artifact_dir / "resnet50.onnx")
        simplified_op_graph = op_graph[:].simplify()
        self.assert_close(op_graph, simplified_op_graph, threshold=10 * THRESHOLD)

    @unittest.skip("Too expensive.")
    def test_vgg16(self):
        op_graph = parse(artifact_dir / "vgg16.onnx")
        simplified_op_graph = op_graph.simplify()
        self.assert_close(op_graph, simplified_op_graph)


class SimplifierTests(AssertCloseMixin, unittest.TestCase):
    def test_SqueezeConvs_1(self):
        from dnnv.nn.transformers.simplifiers import SqueezeConvs

        input_op = operations.Input((1, 1, 3, 3), np.float32)
        w = np.zeros((1, 1, 1, 1))
        w[0, 0] = 1
        b = np.array([0.5])
        conv_op1 = operations.Conv(input_op, w, b)
        w = np.random.randn(1, 1, 1, 1)
        b = np.random.randn(1)
        conv_op2 = operations.Conv(conv_op1, w, b)

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(
            op_graph,
            simplified_op_graph,
            input_shape=(1, 1, 3, 3),
            threshold=10 * THRESHOLD,
        )

        w = np.random.randn(2, 1, 3, 3)
        b = np.random.randn(2)
        conv_op2 = operations.Conv(conv_op1, w, b)

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(
            op_graph,
            simplified_op_graph,
            input_shape=(1, 1, 3, 3),
            threshold=10 * THRESHOLD,
        )

    def test_SqueezeConvs_2(self):
        from dnnv.nn.transformers.simplifiers import SqueezeConvs

        input_op = operations.Input((1, 3, 3, 3), np.float32)
        w = np.zeros((3, 3, 1, 1))
        for i in range(3):
            w[i, i] = 1
        b = np.array([0.1, 0.5, 1.0])
        conv_op1 = operations.Conv(input_op, w, b)
        w = np.random.randn(1, 3, 3, 3)
        b = np.random.randn(1)
        conv_op2 = operations.Conv(conv_op1, w, b)

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(
            op_graph,
            simplified_op_graph,
            input_shape=(1, 3, 3, 3),
            threshold=10 * THRESHOLD,
        )

        w = np.random.randn(3, 3, 3, 3)
        b = np.random.randn(3)
        conv_op2 = operations.Conv(conv_op1, w, b)

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(
            op_graph,
            simplified_op_graph,
            input_shape=(1, 3, 3, 3),
            threshold=10 * THRESHOLD,
        )

    def test_SqueezeConvs_3(self):
        from dnnv.nn.transformers.simplifiers import SqueezeConvs

        input_op = operations.Input((1, 3, 3, 3), np.float32)
        w = np.zeros((3, 3, 1, 1))
        for i in range(3):
            w[i, i] = 1
        b = np.array([0.1, 0.5, 1.0])
        conv_op1 = operations.Conv(input_op, w, b)
        w = np.random.randn(1, 3, 3, 3)
        b = np.random.randn(1)
        conv_op2 = operations.Conv(conv_op1, w, b, pads=(1, 1, 1, 1))

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 3, 3, 3))

        w = np.random.randn(3, 3, 3, 3)
        b = np.random.randn(3)
        conv_op2 = operations.Conv(conv_op1, w, b, pads=(1, 1, 1, 1))

        op_graph = OperationGraph([conv_op2])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(SqueezeConvs(op_graph_copy))
        )
        self.assert_close(op_graph, simplified_op_graph, input_shape=(1, 3, 3, 3))

    def test_ConvertBatchNorm_1(self):
        from dnnv.nn.transformers.simplifiers import ConvertBatchNorm

        input_op = operations.Input((1, 3, 3, 3), np.float32)
        w = np.zeros((3, 3, 1, 1))
        for i in range(3):
            w[i, i] = 1
        b = np.array([0.1, 0.5, 1.0])
        conv_op1 = operations.Conv(input_op, w, b)
        scale = np.random.randn(3)
        bias = np.random.randn(3)
        mean = np.random.randn(3)
        variance = np.random.random()
        bn_op1 = operations.BatchNormalization(conv_op1, scale, bias, mean, variance)

        op_graph = OperationGraph([bn_op1])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(ConvertBatchNorm(op_graph_copy))
        )
        self.assertNotEqual(
            type(op_graph.output_operations[0]),
            type(simplified_op_graph.output_operations[0]),
        )
        self.assert_close(
            op_graph,
            simplified_op_graph,
            input_shape=(1, 3, 3, 3),
            threshold=10 * THRESHOLD,
        )

    def test_ConvertBatchNorm_2(self):
        from dnnv.nn.transformers.simplifiers import ConvertBatchNorm

        input_op = operations.Input((1, 10), np.float32)
        w = np.random.random(size=(10, 20))
        b = np.random.random(size=(20,))
        gemm_op1 = operations.Gemm(input_op, w, b)
        scale = np.random.randn(20)
        bias = np.random.randn(20)
        mean = np.random.randn(20)
        variance = np.random.random()
        bn_op1 = operations.BatchNormalization(gemm_op1, scale, bias, mean, variance)

        op_graph = OperationGraph([bn_op1])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(ConvertBatchNorm(op_graph_copy))
        )
        self.assertNotEqual(
            type(op_graph.output_operations[0]),
            type(simplified_op_graph.output_operations[0]),
        )
        self.assert_close(
            op_graph, simplified_op_graph, input_shape=(1, 10), threshold=10 * THRESHOLD
        )

        gemm_op1 = operations.Gemm(input_op, w.T, b, transpose_b=True)
        scale = np.random.randn(20)
        bias = np.random.randn(20)
        mean = np.random.randn(20)
        variance = np.random.random()
        bn_op1 = operations.BatchNormalization(gemm_op1, scale, bias, mean, variance)

        op_graph = OperationGraph([bn_op1])
        op_graph_copy = op_graph[:]
        simplified_op_graph = OperationGraph(
            op_graph_copy.walk(ConvertBatchNorm(op_graph_copy))
        )
        self.assertNotEqual(
            type(op_graph.output_operations[0]),
            type(simplified_op_graph.output_operations[0]),
        )
        self.assert_close(
            op_graph, simplified_op_graph, input_shape=(1, 10), threshold=10 * THRESHOLD
        )


class SlicerTests(AssertCloseMixin, unittest.TestCase):
    def assert_composition_equals(
        self, model, slices, threshold=THRESHOLD, input_shape=(1, 3, 224, 224)
    ):
        for i in range(5):
            tf.compat.v1.reset_default_graph()
            x = np.random.normal(size=input_shape).astype(np.float32)
            y_1 = model(x)
            y_2 = (x,)
            for model_slice in slices:
                y_2 = model_slice(*y_2, squeeze=False)
            self.assertEqual(len(y_2), 1)
            y_2 = y_2[0]
            self.assertTrue(
                (np.abs(y_1 - y_2) < threshold).all(),
                f"{np.abs(y_1 - y_2).max()} >= {threshold}",
            )

    def test_a_gt_b(self):
        op_graph = parse(artifact_dir / "a_gt_b.onnx")

        pos_end_index = op_graph[:1]
        neg_end_index = op_graph[:-4]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[:2]
        neg_end_index = op_graph[:-3]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[:3]
        neg_end_index = op_graph[:-2]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[:4]
        neg_end_index = op_graph[:-1]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[1:2]
        neg_end_index = op_graph[1:-3]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-4:2]
        neg_end_index = op_graph[-4:-3]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[2:3]
        neg_end_index = op_graph[2:-2]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-3:3]
        neg_end_index = op_graph[-3:-2]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[3:4]
        neg_end_index = op_graph[3:-1]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-2:4]
        neg_end_index = op_graph[-2:-1]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[:4]
        neg_end_index = op_graph[:-1]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[1:]
        neg_end_index = op_graph[1:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-4:]
        neg_end_index = op_graph[-4:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[2:]
        neg_end_index = op_graph[2:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-3:]
        neg_end_index = op_graph[-3:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[3:]
        neg_end_index = op_graph[3:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

        pos_end_index = op_graph[-2:]
        neg_end_index = op_graph[-2:]
        self.assert_close(pos_end_index, neg_end_index, input_shape=(1, 2))

    def test_resnet34(self):
        op_graph = parse(artifact_dir / "resnet34.onnx")

        first_half = op_graph[:60]
        self.assertEqual(
            len(first_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            first_half.output_operations[0],
            operations.Add,
            "Expected first slice to end with Add.",
        )
        second_half = op_graph[60:]
        self.assertEqual(
            len(second_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            second_half.output_operations[0],
            operations.Gemm,
            "Expected first slice to end with Gemm.",
        )

        self.assert_composition_equals(op_graph, (first_half, second_half))

    def test_resnet50(self):
        op_graph = parse(artifact_dir / "resnet50.onnx")

        first_half = op_graph[:84]
        self.assertEqual(
            len(first_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            first_half.output_operations[0],
            operations.Add,
            "Expected first slice to end with Add.",
        )
        second_half = op_graph[84:]
        self.assertEqual(
            len(second_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            second_half.output_operations[0],
            operations.Gemm,
            "Expected first slice to end with Gemm.",
        )

        self.assert_composition_equals(op_graph, (first_half, second_half))

    def test_resnet50_multi_output(self):
        op_graph = parse(artifact_dir / "resnet50.onnx")

        first_half = op_graph[:83]
        self.assertEqual(
            len(first_half.output_operations), 2, "Expected 2 output operations."
        )
        self.assertIsInstance(
            first_half.output_operations[0],
            operations.BatchNormalization,
            "Expected first output operation to be BatchNormalization.",
        )
        self.assertIsInstance(
            first_half.output_operations[1],
            operations.BatchNormalization,
            "Expected second output operation to be BatchNormalization.",
        )
        second_half = op_graph[84:]
        self.assertEqual(
            len(second_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            second_half.output_operations[0],
            operations.Gemm,
            "Expected first slice to end with Gemm.",
        )

        self.assert_composition_equals(
            op_graph,
            (
                first_half,
                OperationGraph(
                    [
                        operations.Add(
                            operations.Input((-1, 1024, 14, 14), np.float32),
                            operations.Input((-1, 1024, 14, 14), np.float32),
                        )
                    ]
                ),
                second_half,
            ),
        )

    def test_vgg16(self):
        op_graph = parse(artifact_dir / "vgg16.onnx")

        first_half = op_graph[:19]
        self.assertEqual(
            len(first_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            first_half.output_operations[0],
            operations.Conv,
            "Expected first slice to end with Add.",
        )

        second_half = op_graph[19:]
        self.assertEqual(
            len(second_half.output_operations), 1, "Only 1 output operation expected."
        )
        self.assertIsInstance(
            second_half.output_operations[0],
            operations.Gemm,
            "Expected first slice to end with Gemm.",
        )
        self.assert_composition_equals(
            op_graph, (first_half, second_half), threshold=1.5 * THRESHOLD
        )


if __name__ == "__main__":
    unittest.main()
