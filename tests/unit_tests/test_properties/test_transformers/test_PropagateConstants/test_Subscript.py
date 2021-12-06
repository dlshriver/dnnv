from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Input
from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Subscript_non_concrete():
    transformer = PropagateConstants()

    expr = Subscript(Symbol("x"), Symbol("i"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Subscript)
    assert new_expr.expr1 is expr.expr1
    assert new_expr.expr2 is expr.expr2

    expr = Subscript(Network("N"), Constant(0))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Subscript)


def test_Subscript_concrete():
    transformer = PropagateConstants()

    expr = Subscript(Constant([0, 2, 4, 6, 8, 10]), Constant(5))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 10

    expr = Subscript(Network("N"), Constant(0))
    expr.concretize(N=OperationGraph([Input(None, None)]))
    new_expr = transformer.visit(expr)
    print(new_expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Network)
    assert new_expr.is_concrete
