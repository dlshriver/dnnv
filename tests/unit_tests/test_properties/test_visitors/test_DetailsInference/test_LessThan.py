import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference, DNNVShapeError


def test_LessThan_symbols():
    inference = DetailsInference()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThan(a, b)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.shapes[expr].value == ()
    assert inference.types[expr].value == bool


def test_LessThan_constants():
    inference = DetailsInference()

    a, b = Constant(99), Constant(3)
    expr = LessThan(a, b)
    inference.visit(expr)

    assert inference.shapes[a].is_concrete
    assert inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.shapes[a].value == ()
    assert inference.shapes[b].value == ()
    assert inference.shapes[expr].value == ()

    assert inference.types[a].is_concrete
    assert inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == np.min_scalar_type(99)
    assert inference.types[b].value == np.min_scalar_type(3)
    assert inference.types[expr].value == bool


def test_LessThan_arrays():
    inference = DetailsInference()

    a, b = Constant(np.array([[1, 2, 3]])), Constant(np.random.rand(1, 1, 1, 3))
    expr = LessThan(a, b)
    inference.visit(expr)

    assert inference.shapes[a].is_concrete
    assert inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.shapes[a].value == (1, 3)
    assert inference.shapes[b].value == (1, 1, 1, 3)
    assert inference.shapes[expr].value == ()

    assert inference.types[a].is_concrete
    assert inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == a.value.dtype
    assert inference.types[b].value == b.value.dtype
    assert inference.types[expr].value == bool


def test_LessThan_incompatible_shapes():
    inference = DetailsInference()

    a, b = Constant(np.random.rand(3, 4)), Constant(np.random.rand(2))
    expr = LessThan(a, b)
    with pytest.raises(DNNVShapeError):
        inference.visit(expr)
