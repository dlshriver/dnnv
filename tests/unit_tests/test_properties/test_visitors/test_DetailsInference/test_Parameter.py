from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Parameter_symbolic():
    inference = DetailsInference()

    expr = Parameter("param1", int)
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == int

    expr = Parameter("param2", float, default=9.8)
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == float


def test_Parameter_concretized():
    inference = DetailsInference()

    expr = Parameter("param1", int)
    expr.concretize(74328)
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == int

    expr = Parameter("param2", float, default=9.8)
    expr.concretize(32.2)
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == float
