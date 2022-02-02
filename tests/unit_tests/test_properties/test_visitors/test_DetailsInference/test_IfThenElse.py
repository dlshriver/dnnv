import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference, DNNVShapeError, DNNVTypeError


def test_IfThenElse_symbols():
    inference = DetailsInference()

    c, t, f = Symbol("c"), Symbol("t"), Symbol("f")
    expr = IfThenElse(c, t, f)
    inference.visit(expr)

    assert inference.shapes[c].is_concrete
    assert not inference.shapes[t].is_concrete
    assert not inference.shapes[f].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert inference.types[c].is_concrete
    assert not inference.types[t].is_concrete
    assert not inference.types[f].is_concrete
    assert not inference.types[expr].is_concrete

    assert inference.shapes[c].value == ()
    assert inference.types[c].value == bool


def test_IfThenElse_constant_cond():
    inference = DetailsInference()

    c, t, f = Constant(False), Symbol("t"), Symbol("f")
    expr = IfThenElse(c, t, f)
    inference.visit(expr)

    assert inference.shapes[c].is_concrete
    assert not inference.shapes[t].is_concrete
    assert not inference.shapes[f].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert inference.types[c].is_concrete
    assert not inference.types[t].is_concrete
    assert not inference.types[f].is_concrete
    assert not inference.types[expr].is_concrete

    assert inference.shapes[c].value == ()
    assert inference.types[c].value == bool


def test_IfThenElse_constant_true_expr():
    inference = DetailsInference()

    c, t, f = Symbol("c"), Constant(np.array((1, 2))), Symbol("f")
    expr = IfThenElse(c, t, f)
    inference.visit(expr)

    assert inference.shapes[c].is_concrete
    assert inference.shapes[t].is_concrete
    assert inference.shapes[f].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.types[c].is_concrete
    assert inference.types[t].is_concrete
    assert not inference.types[f].is_concrete
    assert not inference.types[expr].is_concrete

    assert inference.shapes[c].value == ()
    assert inference.types[c].value == bool
    assert inference.shapes[t].value == t.value.shape
    assert inference.types[t].value == t.value.dtype
    assert inference.shapes[f].value == t.value.shape
    assert inference.shapes[expr].value == t.value.shape


def test_IfThenElse_constant_false_expr():
    inference = DetailsInference()

    c, t, f = Symbol("c"), Symbol("t"), Constant(np.array((1, 2)))
    expr = IfThenElse(c, t, f)
    inference.visit(expr)

    assert inference.shapes[c].is_concrete
    assert inference.shapes[t].is_concrete
    assert inference.shapes[f].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.types[c].is_concrete
    assert not inference.types[t].is_concrete
    assert inference.types[f].is_concrete
    assert not inference.types[expr].is_concrete

    assert inference.shapes[c].value == ()
    assert inference.types[c].value == bool
    assert inference.shapes[t].value == f.value.shape
    assert inference.shapes[f].value == f.value.shape
    assert inference.types[f].value == f.value.dtype
    assert inference.shapes[expr].value == f.value.shape


def test_IfThenElse_constant_true_false_expr():
    inference = DetailsInference()

    c, t, f = Symbol("c"), Constant(1), Constant(np.array(1.0))
    expr = IfThenElse(c, t, f)
    inference.visit(expr)

    assert inference.shapes[c].is_concrete
    assert inference.shapes[t].is_concrete
    assert inference.shapes[f].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.types[c].is_concrete
    assert inference.types[t].is_concrete
    assert inference.types[f].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.shapes[c].value == ()
    assert inference.types[c].value == bool
    assert inference.shapes[t].value == ()
    assert inference.types[t].value == np.min_scalar_type(t.value)
    assert inference.shapes[f].value == f.value.shape
    assert inference.types[f].value == f.value.dtype
    assert inference.shapes[expr].value == f.value.shape
    assert inference.types[expr].value == np.result_type(
        np.min_scalar_type(t.value), f.value.dtype
    )


def test_IfThenElse_incompatible_shapes():
    inference = DetailsInference()

    with get_context():
        c, t, f = (
            Symbol("c"),
            Constant(np.random.rand(3, 5)),
            Constant(np.random.rand(1, 2)),
        )
        expr = IfThenElse(c, t, f)
        with pytest.raises(DNNVShapeError):
            inference.visit(expr)

    with get_context():
        c, t, f = Constant(np.random.rand(3, 5) > 0.5), Symbol("true"), Symbol("false")
        expr = IfThenElse(c, t, f)
        with pytest.raises(DNNVShapeError):
            inference.visit(expr)


def test_IfThenElse_incompatible_types():
    inference = DetailsInference()

    with get_context():
        c, t, f = Constant(8), Symbol("true"), Symbol("false")
        expr = IfThenElse(c, t, f)
        with pytest.raises(DNNVTypeError):
            inference.visit(expr)
