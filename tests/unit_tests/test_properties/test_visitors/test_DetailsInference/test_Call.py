import numpy as np
import pytest

from dnnv.nn import operations
from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *
from dnnv.properties.visitors import (
    DetailsInference,
    DNNVInferenceError,
    DNNVShapeError,
    DNNVTypeError,
)


def test_Call_symbolic_no_args():
    inference = DetailsInference()

    f = Symbol("f")
    expr = f()
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Call_symbolic_args():
    inference = DetailsInference()

    f = Symbol("f")
    args = [Symbol(f"a{i}") for i in range(3)]
    kwargs = {f"kw{i}": Symbol(f"kw{i}") for i in range(2)}
    expr = f(*args, **kwargs)
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Call_constant_no_args():
    inference = DetailsInference()

    f = Constant(lambda: 100)
    expr = f()
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == np.min_scalar_type(100)


def test_Call_constant_args():
    inference = DetailsInference()

    f = Constant(lambda x, y: x + y)
    expr = f(Constant(100), Constant(1000))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == np.min_scalar_type(1000)


def test_Call_network_symbolic():
    inference = DetailsInference()

    N = Network("N")
    x = Symbol("x")
    expr = N(x)
    inference.visit(expr)

    assert not inference.shapes[x].is_concrete
    assert not inference.shapes[expr].is_concrete
    assert not inference.types[x].is_concrete
    assert not inference.types[expr].is_concrete


def test_Call_network_concrete():
    inference = DetailsInference()

    op_graph = OperationGraph(
        [
            operations.Gemm(
                operations.Input((-1, 5), np.float32),
                np.random.randn(5, 3).astype(np.float32),
                np.zeros(3, np.float32),
            )
        ]
    )

    N = Network("N")
    N.concretize(op_graph)
    x = Symbol("x")
    expr = N(x)
    inference.visit(expr)

    assert inference.shapes[x].is_concrete
    assert inference.shapes[expr].is_concrete
    assert inference.types[x].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.shapes[x].value == (1, 5)
    assert inference.shapes[expr].value == (1, 3)
    assert inference.types[x].value == np.float32
    assert inference.types[expr].value == np.float32


def test_Call_network_multiple_outputs():
    inference = DetailsInference()

    input_op = operations.Input((-1, 5), np.float32)
    op_graph = OperationGraph(
        [
            operations.Gemm(
                input_op,
                np.random.randn(5, 3).astype(np.float32),
                np.zeros(3, np.float32),
            ),
            input_op,
        ]
    )

    N = Network("N")
    N.concretize(op_graph)
    x = Symbol("x")
    expr = N(x)
    with pytest.raises(DNNVInferenceError):
        inference.visit(expr)


def test_Call_incompatible_shapes():
    inference = DetailsInference()

    op_graph = OperationGraph(
        [
            operations.Gemm(
                operations.Input((-1, 5), np.float32),
                np.random.randn(5, 3).astype(np.float32),
                np.zeros(3, np.float32),
            )
        ]
    )

    N = Network("N")
    N.concretize(op_graph)
    x = Constant(np.ones((1, 8), np.float32))
    expr = N(x)
    with pytest.raises(DNNVShapeError):
        inference.visit(expr)


def test_Call_incompatible_types():
    inference = DetailsInference()

    op_graph = OperationGraph(
        [
            operations.Gemm(
                operations.Input((-1, 5), np.float32),
                np.random.randn(5, 3).astype(np.float32),
                np.zeros(3, np.float32),
            )
        ]
    )

    N = Network("N")
    N.concretize(op_graph)
    x = Constant(np.ones((1, 5), np.float64))
    expr = N(x)
    with pytest.raises(DNNVTypeError):
        inference.visit(expr)
