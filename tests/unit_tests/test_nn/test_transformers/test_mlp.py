import numpy as np
import pytest

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.mlp import MLPTransformer
from dnnv.nn.visitors import EnsureSupportVisitor


def test_mlp_errors():
    input_op = operations.Input(np.array([21, 5]), np.dtype(np.float32))
    op_graph = OperationGraph([input_op])

    # weird batch size
    with pytest.raises(NotImplementedError):
        _ = MLPTransformer().transform(op_graph)

    input_op_1 = operations.Input(np.array([1, 5]), np.dtype(np.float32))
    input_op_2 = operations.Input(np.array([1, 5]), np.dtype(np.int32))
    add_op = operations.Add(input_op_1, input_op_2)
    op_graph = OperationGraph([add_op])

    # inconsistent data types
    with pytest.raises(NotImplementedError):
        _ = MLPTransformer().transform(op_graph)

    input_op = operations.Input(np.array([1, 5]), np.dtype(np.float32))
    gen_op = operations.RandomUniform(
        np.array([1, 5]), dtype=np.dtype(np.int32), low=0.0, high=1.0
    )
    add_op = operations.Add(input_op, gen_op)
    op_graph = OperationGraph([add_op])

    # inconsistent data types
    with pytest.raises(NotImplementedError):
        _ = MLPTransformer().transform(op_graph)


def test_mlp_single_input_multiple_paths():
    input_op = operations.Input(np.array([-1, 5]), np.dtype(np.float32))
    mul_op_1 = operations.Mul(input_op, np.float32(4.0))
    mul_op_2 = operations.Mul(input_op, np.float32(2.0))
    add_op = operations.Add(mul_op_1, mul_op_2)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x = np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
        y1 = op_graph(x)
        y2 = mlp_op_graph(x)
        assert np.allclose(y1, y2)


def test_mlp_multiple_inputs():
    input_op_1 = operations.Input(np.array([-1, 5]), np.dtype(np.float32))
    gemm_op_1 = operations.Gemm(
        input_op_1,
        np.random.randn(5, 7).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    relu_op_1 = operations.Relu(gemm_op_1)

    input_op_2 = operations.Input(np.array([-1, 3]), np.dtype(np.float32))
    gemm_op_2 = operations.Gemm(
        input_op_2,
        np.random.randn(3, 7).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    relu_op_2 = operations.Relu(gemm_op_2)

    add_op = operations.Add(relu_op_1, relu_op_2)
    gemm_op_3 = operations.Gemm(
        add_op,
        np.random.randn(7, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )
    op_graph = OperationGraph([gemm_op_3])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_op_1.shape[1:]).astype(input_op_1.dtype)
        x2 = np.random.randn(1, *input_op_2.shape[1:]).astype(input_op_2.dtype)
        y1 = op_graph(x1, x2)
        y2 = mlp_op_graph(np.hstack([x1, x2]))
        assert np.allclose(y1, y2)


def test_mlp_conv():
    input_op = operations.Input(np.array([-1, 2, 5, 5]), np.dtype(np.float32))
    conv_op = operations.Conv(
        input_op,
        np.random.randn(1, 2, 2, 2).astype(np.float32),
        np.random.randn(1).astype(np.float32),
    )
    relu_op = operations.Relu(conv_op)
    flatten_op = operations.Flatten(relu_op)
    gemm_op = operations.Gemm(
        flatten_op,
        np.random.randn((5 - 1) ** 2, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )

    op_graph = OperationGraph([gemm_op])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x = np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
        y1 = op_graph(x)
        y2 = mlp_op_graph(x.reshape(x.shape[0], -1))
        assert np.allclose(y1, y2)


def test_mlp_randomuniform():
    input_op = operations.Input((-1, 10), np.dtype(np.float32))
    generator_op = operations.RandomUniform(
        (1, 10), dtype=np.dtype(np.float32), low=-15.0, high=322.0
    )
    add_op = operations.Add(input_op, generator_op)

    op_graph = OperationGraph([add_op])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    input_op = operations.Input((-1, 10), np.dtype(np.float32))
    generator_op = operations.Input((-1, 10), np.dtype(np.float32))
    add_op = operations.Add(input_op, generator_op)
    nonrandom_op_graph = OperationGraph([add_op])

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
        x2 = np.random.rand(1, *generator_op.shape[1:]).astype(generator_op.dtype)
        y1 = nonrandom_op_graph(x1, x2 * (322 + 15) - 15)
        y2 = mlp_op_graph(np.hstack([x1, x2]))
        assert np.allclose(y1, y2)


def test_mlp_randomuniform_keepgenerators():
    input_op = operations.Input((1, 10), np.dtype(np.float32))
    generator_op = operations.RandomUniform(
        (1, 10), dtype=np.dtype(np.float32), low=0.0, high=1.0, seed=5.0
    )
    add_op = operations.Add(input_op, generator_op)

    op_graph = OperationGraph([add_op])

    mlp_op_graph = MLPTransformer().transform(op_graph, keep_generators=True)

    assert EnsureSupportVisitor(
        [
            operations.Input,
            operations.Gemm,
            operations.Relu,
            operations.RandomUniform,
            operations.Concat,
        ]
    ).walk(mlp_op_graph)
    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph[2:])

    # set generator seed
    assert len(mlp_op_graph.generators) == 1
    mlp_op_graph.generators[0].seed = 5.0

    for _ in range(10):
        x1 = np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
        y1 = op_graph(x1)
        y2 = mlp_op_graph(x1)
        assert np.allclose(y1, y2)


def test_mlp_multiple_path_joins():
    input_op_1 = operations.Input(np.array([-1, 5]), np.dtype(np.float32))
    gemm_op_1 = operations.Gemm(
        input_op_1,
        np.random.randn(5, 7).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    relu_op_1 = operations.Relu(gemm_op_1)

    input_op_2 = operations.Input(np.array([-1, 3]), np.dtype(np.float32))
    gemm_op_2 = operations.Gemm(
        input_op_2,
        np.random.randn(3, 7).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    relu_op_2 = operations.Relu(gemm_op_2)

    add_op = operations.Add(relu_op_1, relu_op_2)

    input_op_3 = operations.Input(np.array([-1, 6]), np.dtype(np.float32))
    gemm_op_3 = operations.Gemm(
        input_op_3,
        np.random.randn(6, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )
    relu_op_3 = operations.Relu(gemm_op_3)

    gemm_op_4 = operations.Gemm(
        add_op,
        np.random.randn(7, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )
    concat_op = operations.Concat([gemm_op_4, relu_op_3], axis=1)

    op_graph = OperationGraph([concat_op])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_op_1.shape[1:]).astype(input_op_1.dtype)
        x2 = np.random.randn(1, *input_op_2.shape[1:]).astype(input_op_2.dtype)
        x3 = np.random.randn(1, *input_op_3.shape[1:]).astype(input_op_3.dtype)
        y1 = op_graph(x1, x2, x3)
        y2 = mlp_op_graph(np.hstack([x1, x2, x3]))
        assert np.allclose(y1, y2)


def test_mlp_multiple_path_joins_with_randomuniform():
    input_op_1 = operations.Input(np.array([-1, 5]), np.dtype(np.float32))
    gemm_op_1 = operations.Gemm(
        input_op_1,
        np.random.randn(5, 7).astype(np.float32),
        np.random.randn(7).astype(np.float32),
    )
    relu_op_1 = operations.Relu(gemm_op_1)

    rand_op_1 = operations.RandomUniform(
        np.array([1, 7]), dtype=np.dtype(np.float32), low=0.0, high=1.0, seed=11.0
    )

    add_op_1 = operations.Add(relu_op_1, rand_op_1)

    rand_op_2 = operations.RandomUniform(
        np.array([1, 6]), dtype=np.dtype(np.float32), low=0.0, high=1.0, seed=13.0
    )
    gemm_op_3 = operations.Gemm(
        rand_op_2,
        np.random.randn(6, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )
    relu_op_3 = operations.Relu(gemm_op_3)

    gemm_op_4 = operations.Gemm(
        add_op_1,
        np.random.randn(7, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    )
    add_op_2 = operations.Add(gemm_op_4, relu_op_3)

    op_graph = OperationGraph([add_op_2])

    mlp_op_graph = MLPTransformer().transform(op_graph)

    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_op_1.shape[1:]).astype(input_op_1.dtype)
        x2 = OperationGraph([rand_op_1])()
        x3 = OperationGraph([rand_op_2])()
        y1 = op_graph(x1)
        y2 = mlp_op_graph(np.hstack([x1, x2, x3]))
        assert np.allclose(y1, y2)


def test_mlp_residual():
    randn_f32 = lambda *shape: np.random.randn(*shape).astype(np.float32)
    input_0 = operations.Input((1, 6), np.float32)
    gemm_0 = operations.Gemm(input_0, randn_f32(6, 5), randn_f32(5))
    sub_0 = operations.Sub(gemm_0, randn_f32(1, 5))
    gemm_1 = operations.Gemm(sub_0, randn_f32(20, 5), randn_f32(20), transpose_b=True)
    relu_0 = operations.Relu(gemm_1)
    gemm_6 = operations.Gemm(relu_0, randn_f32(1, 20), randn_f32(1), transpose_b=True)
    concat_0 = operations.Concat([input_0, gemm_6], axis=1)

    op_graph = OperationGraph([concat_0])

    mlp_op_graph = MLPTransformer().transform(op_graph)
    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_0.shape[1:]).astype(input_0.dtype)
        y1 = op_graph(x1)
        y2 = mlp_op_graph(x1)
        assert np.allclose(y1, y2)


def test_mlp_random_residual():
    randn_f32 = lambda *shape: np.random.randn(*shape).astype(np.float32)
    input_0 = operations.Input((1, 6), np.float32)
    gemm_0 = operations.Gemm(input_0, randn_f32(6, 5), randn_f32(5))
    sub_0 = operations.Sub(gemm_0, randn_f32(1, 5))
    gemm_1 = operations.Gemm(sub_0, randn_f32(20, 5), randn_f32(20), transpose_b=True)
    relu_0 = operations.Relu(gemm_1)
    gemm_6 = operations.Gemm(relu_0, randn_f32(1, 20), randn_f32(1), transpose_b=True)
    rand_0 = operations.RandomUniform(
        (1, 6), dtype=np.float32, low=0.0, high=1.0, seed=37.0
    )
    concat_0 = operations.Concat([input_0, gemm_6, rand_0], axis=1)

    op_graph = OperationGraph([concat_0])

    mlp_op_graph = MLPTransformer().transform(op_graph)
    assert EnsureSupportVisitor(
        [operations.Input, operations.Gemm, operations.Relu]
    ).walk(mlp_op_graph)

    for _ in range(10):
        x1 = np.random.randn(1, *input_0.shape[1:]).astype(input_0.dtype)
        x2 = OperationGraph([rand_0])()
        y1 = op_graph(x1)
        y2 = mlp_op_graph(np.hstack([x1, x2]))
        assert np.allclose(y1, y2)
