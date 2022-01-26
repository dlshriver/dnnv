from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Attribute_symbolic():
    inference = DetailsInference()

    a, b = Symbol("a"), Symbol("b")
    expr = Attribute(a, b)
    inference.visit(expr)

    assert not inference.shapes[a].is_concrete
    assert not inference.shapes[b].is_concrete
    assert not inference.shapes[expr].is_concrete

    assert not inference.types[a].is_concrete
    assert not inference.types[b].is_concrete
    assert not inference.types[expr].is_concrete


def test_Attribute_constants():
    inference = DetailsInference()

    class Test:
        pass

    test = Test()
    test.attribute = (1, 2, 3)

    a, b = Constant(test), Constant("attribute")
    expr = Attribute(a, b)
    inference.visit(expr)

    assert inference.shapes[a].is_concrete
    assert inference.shapes[b].is_concrete
    assert inference.shapes[expr].is_concrete

    assert inference.shapes[a].value == ()
    assert inference.shapes[b].value == ()
    assert inference.shapes[expr].value == (3,)

    assert inference.types[a].is_concrete
    assert inference.types[b].is_concrete
    assert inference.types[expr].is_concrete

    assert inference.types[a].value == Test
    assert inference.types[b].value == str
    assert inference.types[expr].value == tuple
