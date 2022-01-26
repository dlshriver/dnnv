import numpy as np

from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Subscript_symbolic():
    inference = DetailsInference()

    x, i = Symbol("x"), Symbol("i")
    expr = Subscript(x, i)
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Subscript_symbolic_expr():
    inference = DetailsInference()

    x = Symbol("x")
    expr = Subscript(x, Constant(0))
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Subscript_symbolic_index():
    inference = DetailsInference()

    i = Symbol("i")
    expr = Subscript(Constant((1, 2, 3)), i)
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == np.asarray((1, 2, 3)).dtype


def test_Subscript_network():
    inference = DetailsInference()

    N = Network("N")
    expr = Subscript(N, Slice(Constant(None), Constant(None), Constant(None)))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == OperationGraph
