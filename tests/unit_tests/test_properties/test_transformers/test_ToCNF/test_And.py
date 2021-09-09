from dnnv.properties.base import *
from dnnv.properties.transformers import ToCNF


def test_And_empty():
    transformer = ToCNF()

    expr = And()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert new_expr.expressions[0].expressions[0] is Constant(True)


def test_And_of_and():
    transformer = ToCNF()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = And()
    expr.expressions.append(And(a, b))
    expr.expressions.append(And(b, c))
    expr.expressions.append(c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 3
    assert Or(Symbol("a")) in new_expr.expressions
    assert Or(Symbol("b")) in new_expr.expressions
    assert Or(Symbol("c")) in new_expr.expressions


def test_And_single_expr():
    transformer = ToCNF()

    a = Symbol("a")
    expr = And(a)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert Or(Symbol("a")) in new_expr.expressions
