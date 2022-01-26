from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Exists_symbolic():
    inference = DetailsInference()

    a, b = Symbol("a"), Symbol("b")
    expr = Exists(a, a)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[expr].is_concrete

    expr = Exists(a, a == b)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.shapes[expr].value == ()
    assert inference.types[expr].value == bool


def test_Exists_constants():
    inference = DetailsInference()

    a = Symbol("x")
    expr = Exists(a, Constant(True))
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()

    assert not inference.types[a].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == bool
