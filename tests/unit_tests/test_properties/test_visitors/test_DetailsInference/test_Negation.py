import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Negation_symbol():
    inference = DetailsInference()

    expr = Negation(Symbol("a"))
    inference.visit(expr)

    assert not inference.shapes[expr].is_concrete
    assert not inference.types[expr].is_concrete


def test_Negation_constant():
    inference = DetailsInference()

    expr = Negation(Constant(32.2))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == np.min_scalar_type(32.2)
