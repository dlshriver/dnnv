from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Forall_symbolic():
    inference = DetailsInference()

    a, b = Symbol("a"), Symbol("b")
    expr = Forall(a, a)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[expr].is_concrete

    expr = Forall(a, a == b)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.shapes[expr].value == ()
    assert inference.types[expr].value == bool


def test_Forall_constants():
    inference = DetailsInference()

    a = Symbol("x")
    expr = Forall(a, Constant(True))
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert inference.shapes[expr].is_concrete
    assert inference.shapes[expr].value == ()

    assert not inference.types[a].is_concrete
    assert inference.types[expr].is_concrete
    assert inference.types[expr].value == bool
