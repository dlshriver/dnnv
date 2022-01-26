from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_Or_of_symbols():
    transformer = DnfTransformer()

    expr = Or(Symbol("x"), Symbol("y"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    assert And(Symbol("x")) in new_expr.expressions
    assert And(Symbol("y")) in new_expr.expressions


def test_Or_of_ors():
    transformer = DnfTransformer()

    expr = Or()
    expr.expressions.append(Or(Symbol("x"), Symbol("y")))
    expr.expressions.append(Or(Symbol("a"), Symbol("b")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 4
    assert And(Symbol("x")) in new_expr.expressions
    assert And(Symbol("y")) in new_expr.expressions
    assert And(Symbol("a")) in new_expr.expressions
    assert And(Symbol("b")) in new_expr.expressions


def test_Or_empty():
    transformer = DnfTransformer()

    expr = Or()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(Or(And(Constant(False))))


def test_Or_and():
    transformer = DnfTransformer()

    expr = Or(And(Symbol("a")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 1
    assert new_expr.expressions[0].expressions[0] is Symbol("a")


def test_Or_and_multi_expressions():
    transformer = DnfTransformer()

    expr = Or(And(Symbol("a"), Symbol("b")), Symbol("c"), Symbol("d"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 3
    assert And(Symbol("a"), Symbol("b")) in new_expr.expressions
    assert And(Symbol("c")) in new_expr.expressions
    assert And(Symbol("d")) in new_expr.expressions


def test_Or_multi_and():
    transformer = DnfTransformer()

    a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
    expr = Or(And(a, b), And(c, d))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2

    assert And(a, b) in new_expr.expressions
    assert And(c, d) in new_expr.expressions


def test_Or_symbols():
    transformer = DnfTransformer()

    a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
    expr = Or(a, b, c, d)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 4

    assert And(a) in new_expr.expressions
    assert And(b) in new_expr.expressions
    assert And(c) in new_expr.expressions
    assert And(d) in new_expr.expressions
