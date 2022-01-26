import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference, DNNVShapeError, DNNVTypeError


def test_And_symbols():
    inference = DetailsInference()

    a, b = Symbol("a"), Symbol("b")
    expr = And(a, b)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[b].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert inference.types[a].is_concrete
    assert inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == bool
    assert inference.types[b].value == bool
    assert inference.types[expr].value == bool


def test_And_constants():
    inference = DetailsInference()

    a, b = Constant(True), Constant(False)
    expr = And(a, b)
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

    assert inference.types[a].value == bool
    assert inference.types[b].value == bool
    assert inference.types[expr].value == bool


def test_And_arrays():
    inference = DetailsInference()

    a, b = Constant(np.random.rand(3) > 0.5), Constant(np.random.rand(1, 1, 1, 3) > 0.5)
    expr = And(a, b)
    inference.visit(expr)

    assert inference.shapes[a].is_concrete
    assert inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.shapes[a].value == (3,)
    assert inference.shapes[b].value == (1, 1, 1, 3)
    assert inference.shapes[expr].value == (1, 1, 1, 3)

    assert inference.types[a].is_concrete
    assert inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == bool
    assert inference.types[b].value == bool
    assert inference.types[expr].value == bool


def test_And_incompatible_shapes():
    inference = DetailsInference()

    a, b = Constant(np.random.rand(3, 4) > 0.5), Constant(np.random.rand(2) > 0.5)
    expr = And(a, b)
    with pytest.raises(DNNVShapeError):
        inference.visit(expr)


def test_And_incompatible_types():
    inference = DetailsInference()

    a, b = Constant(True), Constant(11)
    expr = And(a, b)
    with pytest.raises(DNNVTypeError):
        inference.visit(expr)
