from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Slice():
    inference = DetailsInference()

    expr = Slice(Constant(None), Constant(None), Constant(None))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == slice
