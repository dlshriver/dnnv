from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Symbol():
    inference = DetailsInference()
    a = Symbol("a")
    inference.visit(a)
    assert not inference.shapes[a].is_concrete
    assert not inference.types[a].is_concrete
