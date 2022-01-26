from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Compose_networks():
    inference = DetailsInference()
    N1 = Network("N1")
    N2 = Network("N2")
    expr = Compose(N1, N2)
    inference.visit(expr)
    assert inference.shapes[expr].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].value == OperationGraph


def test_Compose_functions():
    inference = DetailsInference()
    f1 = Constant(max)
    f2 = Constant(abs)
    expr = Compose(f1, f2)
    inference.visit(expr)
    assert inference.shapes[expr].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].value == type(max)
