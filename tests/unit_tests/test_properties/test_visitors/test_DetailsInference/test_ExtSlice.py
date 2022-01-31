from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_ExtSlice():
    inference = DetailsInference()

    expr = ExtSlice(slice(1, -1), Constant(1))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == tuple
