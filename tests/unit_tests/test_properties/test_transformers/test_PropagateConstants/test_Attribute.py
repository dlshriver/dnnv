from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Attribute():
    transformer = PropagateConstants()

    expr = Attribute(Symbol("string"), Constant("isalpha"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Attribute)
    assert new_expr.expr1 is Symbol("string")
    assert new_expr.expr2 is Constant("isalpha")

    expr = Attribute(Constant("test"), Constant("isalpha"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "test".isalpha

    N = Network("N")
    fake_network = lambda x: x
    fake_network.compose = lambda f, g: lambda y: f(g(y))
    N.concretize(fake_network)
    expr = Attribute(N, Constant("compose"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == N.compose
