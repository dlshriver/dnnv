from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference


def test_Network():
    inference = DetailsInference()
    N = Network("N")
    inference.visit(N)
    assert inference.shapes[N].is_concrete
    assert inference.types[N].is_concrete
    assert inference.shapes[N].value == ()
    assert inference.types[N].value == OperationGraph
