import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference, DNNVTypeError


def test_Not_symbol():
    inference = DetailsInference()

    a = Symbol("a")
    expr = Not(a)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert inference.types[a].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == bool
    assert inference.types[expr].value == bool


def test_Not_constant():
    inference = DetailsInference()

    expr = Not(Constant(True))
    inference.visit(expr)

    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == bool


def test_Not_arrays():
    inference = DetailsInference()

    x = Constant(np.random.rand(1, 3) > 0.5)
    expr = Not(x)
    inference.visit(expr)

    assert inference.shapes[x].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.shapes[x].value == (1, 3)
    assert inference.shapes[expr].value == (1, 3)

    assert inference.types[x].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[x].value == bool
    assert inference.types[expr].value == bool


def test_Not_incompatible_type():
    inference = DetailsInference()

    a = Constant(np.random.rand(3, 4))
    expr = Not(a)
    with pytest.raises(DNNVTypeError):
        inference.visit(expr)
