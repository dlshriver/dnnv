from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_And_empty():
    transformer = DnfTransformer()
    transformer._top_level = False

    expr = And()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(True)


def test_And_of_and():
    transformer = DnfTransformer()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = And()
    expr.expressions.append(And(a, b))
    expr.expressions.append(And(b, c))
    expr.expressions.append(c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 3
    assert Symbol("a") in new_expr.expressions[0].expressions
    assert Symbol("b") in new_expr.expressions[0].expressions
    assert Symbol("c") in new_expr.expressions[0].expressions


def test_And_single_expr():
    transformer = DnfTransformer()

    a = Symbol("a")
    expr = And(a)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(Symbol("a")) in new_expr.expressions


def test_And_multi_expr():
    transformer = DnfTransformer()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = And(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(Symbol("a"), Symbol("b"), Symbol("c")) in new_expr.expressions


def test_And_of_or():
    transformer = DnfTransformer()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = And(Or(a, b), Or(b, c))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 4
    assert And(a, b) in new_expr.expressions
    assert And(a, c) in new_expr.expressions
    assert And(b) in new_expr.expressions
    assert And(b, c) in new_expr.expressions


def test_And_of_mixed():
    transformer = DnfTransformer()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    d, e, f = Symbol("d"), Symbol("e"), Symbol("f")
    expr = And(Or(a, b), Or(b, c), d, e, f)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 4
    assert And(a, b, d, e, f) in new_expr.expressions
    assert And(a, c, d, e, f) in new_expr.expressions
    assert And(b, d, e, f) in new_expr.expressions
    assert And(b, c, d, e, f) in new_expr.expressions
