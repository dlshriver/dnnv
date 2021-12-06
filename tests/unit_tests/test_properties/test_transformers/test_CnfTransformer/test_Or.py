from dnnv.properties import *
from dnnv.properties.transformers import CnfTransformer


def test_Or_of_symbols():
    transformer = CnfTransformer()

    expr = Or(Symbol("x"), Symbol("y"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    new_expr_ = list(new_expr.expressions)[0]
    assert isinstance(new_expr_, Or)
    assert len(new_expr_.expressions) == 2
    assert Symbol("x") in new_expr_.expressions
    assert Symbol("y") in new_expr_.expressions


def test_Or_of_ors():
    transformer = CnfTransformer()

    expr = Or()
    expr.expressions.append(Or(Symbol("x"), Symbol("y")))
    expr.expressions.append(Or(Symbol("a"), Symbol("b")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    new_expr_ = list(new_expr.expressions)[0]
    assert isinstance(new_expr_, Or)
    assert len(new_expr_.expressions) == 4
    assert Symbol("x") in new_expr_.expressions
    assert Symbol("y") in new_expr_.expressions
    assert Symbol("a") in new_expr_.expressions
    assert Symbol("b") in new_expr_.expressions


def test_Or_empty():
    transformer = CnfTransformer()
    transformer._top_level = False

    expr = Or()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert new_expr.expressions[0].expressions[0] is Constant(False)


def test_Or_and():
    transformer = CnfTransformer()

    expr = Or(And(Symbol("a")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert new_expr.expressions[0].expressions[0] is Symbol("a")


def test_Or_and_multi_expressions():
    transformer = CnfTransformer()

    expr = Or(And(Symbol("a"), Symbol("b")), Symbol("c"), Symbol("d"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 3
    assert Symbol("c") in new_expr.expressions[0].expressions
    assert Symbol("d") in new_expr.expressions[0].expressions
    assert isinstance(new_expr.expressions[1], Or)
    assert len(new_expr.expressions[1].expressions) == 3
    assert Symbol("c") in new_expr.expressions[1].expressions
    assert Symbol("d") in new_expr.expressions[1].expressions

    assert (
        Symbol("a") in new_expr.expressions[0].expressions
        or Symbol("a") in new_expr.expressions[1].expressions
    )
    assert (
        Symbol("b") in new_expr.expressions[0].expressions
        or Symbol("b") in new_expr.expressions[1].expressions
    )


def test_Or_multi_and():
    transformer = CnfTransformer()

    a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
    expr = Or(And(a, b), And(c, d))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 4
    assert isinstance(new_expr.expressions[0], Or)
    assert isinstance(new_expr.expressions[1], Or)
    assert isinstance(new_expr.expressions[2], Or)
    assert isinstance(new_expr.expressions[3], Or)

    assert Or(a, c) in new_expr.expressions
    assert Or(a, d) in new_expr.expressions
    assert Or(b, c) in new_expr.expressions
    assert Or(b, d) in new_expr.expressions
